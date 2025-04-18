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

def generate_question(client, transcription, num_qas_per_transcription, model="gpt-4o") -> list:
    """Generate multiple-choice question(s) based on the transcription"""
    
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
    """

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_list",
            "strict": True,
            "description": "Question and Answer list",
            "schema": {
                "type": "object",
                "properties": {
                    "qas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question text"
                                },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "An answer option"
                                    }
                                },
                                "correct_answer": {
                                    "type": "string",
                                    "description": "The letter corresponding to the correct answer"
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Detailed explanation of why the correct answer is right and others are wrong"
                                }
                            },
                            "required": ["question", "options", "correct_answer", "explanation"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["qas"],
                "additionalProperties": False
            }
        }
    }
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format=response_format,
            messages=[
                {"role": "system", "content": "You are a medical education expert who creates high-quality assessment questions for healthcare professionals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return json.loads(response.choices[0].message.content)["qas"]
        # return response.choices[0].message.parsed
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate multiple-choice questions from medical transcriptions")
    parser.add_argument("--input", default="data/mtsamples.csv", help="Path to input CSV file")
    parser.add_argument("--output", default="data/generated_questions.json", help="Path to output JSON file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--medical_specialty", default="Cardiovascular / Pulmonary", help="Medical specialty to filter to")
    parser.add_argument("--transcriptions", type=int, default=10, help="Number of transcriptions to sample")
    parser.add_argument("--samples", type=int, default=1, help="Number of question-answer pairs to sample per transcription")
    
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
        print(f"Sampling {args.transcriptions} transcriptions")
        df = df.sample(args.transcriptions, random_state=42)
    
    # Generate questions
    questions = []
    print(f"Generating {args.samples} questions per transcription using {args.model}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isna(row['transcription']) and len(str(row['transcription'])) > 100:
            qas = generate_question(
                client, 
                transcription=str(row['transcription']), 
                num_qas_per_transcription=args.samples, 
                model=args.model
            )
            if qas:
                for qa in qas:
                    # Add metadata if available in the dataset
                    if 'medical_specialty' in df.columns:
                        qa['medical_specialty'] = row.get('medical_specialty', '')
                    if 'description' in df.columns:
                        qa['description'] = row.get('description', '')
                    
                    # Add the original transcription in
                    qa["transcription"] = str(row["transcription"])
                    
                    questions.append(qa)
                
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
    
    # Save questions to output file
    with open(args.output, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"Successfully generated {len(questions)} questions and saved to {args.output}")

if __name__ == "__main__":
    main()