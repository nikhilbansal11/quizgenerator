from typing import List, Dict, Optional
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = "AIzaSyBvEWR0peDVveZ-0VX2f1KDdQ6giMKUZjw"

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correctOption: str

class QuizRequest(BaseModel):
    prompt: str
    text: str
    num_questions: int
    default_prompt: Optional[str] = None

def extract_json_from_response(response_text: str) -> List[Dict]:
    """
    Extract JSON data from Gemini's response text which might be wrapped in markdown code blocks.
    
    Args:
        response_text (str): Raw response text from Gemini
        
    Returns:
        List[Dict]: Parsed JSON data
    """
    # Remove markdown code blocks if present
    clean_text = response_text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {str(e)}")

def generate_quiz(
    prompt: str,
    text: str,
    num_questions: int,
    default_prompt: Optional[str] = "Generate educational quiz questions from the given text."
) -> List[Dict]:
    """
    Generate quiz questions using the Gemini API.
    
    Args:
        prompt (str): User-defined prompt for quiz generation
        text (str): Source text to generate questions from
        num_questions (int): Number of questions to generate
        default_prompt (str, optional): Fallback prompt if none provided
    
    Returns:
        List[Dict]: List of quiz questions with options and correct answers
    """
    
    # Use provided prompt or fall back to default
    final_prompt = prompt or default_prompt
    
    # Construct the complete prompt for Gemini
    system_prompt = f"""
    Based on the following text, generate {num_questions} multiple-choice questions.
    User prompt: {final_prompt}
    
    Format each question as a JSON object with the following structure:
    {{
        "question": "string",
        "options": ["string", "string", "string", "string"],
        "correctOption": "string"
    }}
    
    Return the questions as a JSON array.
    
    Source text:
    {text}
    """
    
    try:
        # Generate response from Gemini
        response = model.generate_content(system_prompt)
        
        # Extract JSON from response text
        quiz_data = extract_json_from_response(response.text)
        
        # Validate response structure
        for question in quiz_data:
            if not all(key in question for key in ['question', 'options', 'correctOption']):
                raise ValueError("Invalid question format in API response")
            if len(question['options']) != 4:
                raise ValueError("Each question must have exactly 4 options")
            if question['correctOption'] not in question['options']:
                raise ValueError("Correct option must be one of the provided options")
        
        return quiz_data
    
    except Exception as e:
        raise Exception(f"Error generating quiz: {str(e)}")

# FastAPI Implementation
app = FastAPI(title="Quiz Generator API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv('ALLOWED_ORIGINS', '["*"]')),  # Load from env or allow all
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]  # Expose all headers
)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key = "AIzaSyBvEWR0peDVveZ-0VX2f1KDdQ6giMKUZjw"):
    # if api_key != os.getenv('API_KEY'):
    #     raise HTTPException(
    #         status_code=401,
    #         detail="Invalid API Key"
    #     )
    
    api_key = "AIzaSyBvEWR0peDVveZ-0VX2f1KDdQ6giMKUZjw"
    return api_key

@app.post("/generate-quiz", response_model=List[QuizQuestion])
async def create_quiz(
    request: QuizRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate a quiz based on provided text and parameters.
    """
    try:
        quiz = generate_quiz(
            prompt=request.prompt,
            text=request.text,
            num_questions=request.num_questions,
            default_prompt=request.default_prompt
        )
        return quiz
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
