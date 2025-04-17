import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import argparse
import time

def load_data(file_path):
    """Load medical transcriptions from CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'transcription' not in df.columns:
            raise ValueError("CSV file must contain a 'transcription' column")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def generate_question(client, transcription, num_qas_per_transcription, model="gpt-4o"):
    """Generate a multiple-choice question based on the transcription using OpenAI API."""
    
    prompt = f"""
    Generate {num_qas_per_transcription} non-trivial multiple-choice question(s) based on the medical transcription below. 
    The question should:
    1. Test clinically relevant knowledge that would be important for healthcare providers
    2. Be specific to information contained in this transcription
    3. Have exactly 4 answer choices (A, B, C, D)
    4. Have only ONE correct answer
    5. Include an explanation for why the correct answer is right and others are wrong
    
    Medical Transcription:
    {transcription}
    
    Format your response as JSON with the following structure:
    {{
        "question": "The question text",
        "options": [
            "A. First option",
            "B. Second option",
            "C. Third option",
            "D. Fourth option"
        ],
        "correct_answer": "A",  # Just the letter of the correct answer
        "explanation": "Detailed explanation of why the correct answer is right and others are wrong"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a medical education expert who creates high-quality assessment questions for healthcare professionals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate multiple-choice questions from medical transcriptions")
    parser.add_argument("--input", default="data/mtsamples.csv", help="Path to input CSV file")
    parser.add_argument("--output", default="data/generated_questions.json", help="Path to output JSON file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--medical_specialty", default="Cardiovascular / Pulmonary", help="Medical specialty to filter to")
    parser.add_argument("--transcriptions", type=int, default=10, help="Number of transcriptions to sample for question generation")
    parser.add_argument("--samples", type=int, default=3, help="Number of question-answer pairs to sample per transcription")
    
    args = parser.parse_args()
    
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)

    # Filter to medical specialty
    df = df.loc[df['medical_specialty'].str.strip() == args.medical_specialty.strip()]
    
    # Sample transcriptions if requested
    if args.transcriptions and args.transcriptions < len(df):
        df = df.sample(args.transcriptions, random_state=42)
    
    # Generate questions
    questions = []
    print(f"Generating questions using {args.model}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isna(row['transcription']) and len(str(row['transcription'])) > 100:
            qa = generate_question(client, str(row['transcription']), args.model)
            if qa:
                # Add metadata if available in the dataset
                if 'medical_specialty' in df.columns:
                    qa['medical_specialty'] = row.get('medical_specialty', '')
                if 'description' in df.columns:
                    qa['description'] = row.get('description', '')
                
                questions.append(qa)
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
    
    # Save questions to output file
    with open(args.output, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"Successfully generated {len(questions)} questions and saved to {args.output}")

if __name__ == "__main__":
    main()