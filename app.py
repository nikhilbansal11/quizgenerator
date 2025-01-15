from typing import List, Dict, Optional
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = "AIzaSyBvEWR0peDVveZ-0VX2f1KDdQ6giMKUZjw"

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

class QuizQuestion:
    def __init__(self, question: str, options: List[str], correctOption: str):
        self.question = question
        self.options = options
        self.correctOption = correctOption
    
    def to_dict(self):
        return {
            "question": self.question,
            "options": self.options,
            "correctOption": self.correctOption
        }

class QuizRequest:
    def __init__(self, prompt: str, text: str, num_questions: int, default_prompt: Optional[str] = None):
        self.prompt = prompt
        self.text = text
        self.num_questions = num_questions
        self.default_prompt = default_prompt or "Generate educational quiz questions from the given text."

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

@app.route('/generate-quiz', methods=['POST'])
def create_quiz():
    """
    Generate a quiz based on provided text and parameters.
    """
    try:
        data = request.json
        quiz_request = QuizRequest(
            prompt=data.get("prompt", ""),
            text=data["text"],
            num_questions=data["num_questions"],
            default_prompt=data.get("default_prompt")
        )
        
        quiz = generate_quiz(
            prompt=quiz_request.prompt,
            text=quiz_request.text,
            num_questions=quiz_request.num_questions,
            default_prompt=quiz_request.default_prompt
        )
        
        # Convert quiz questions to list of QuizQuestion objects
        quiz_questions = [QuizQuestion(q['question'], q['options'], q['correctOption']).to_dict() for q in quiz]
        return jsonify(quiz_questions), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Set Flask app to production-ready (ensure the app is ready for deployment)
    app.run(host='0.0.0.0', port=8000, debug=False)
