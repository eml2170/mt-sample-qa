import json

def get_qas():
    with open("data/generated_questions.json", "r") as f:
        return json.load(f)