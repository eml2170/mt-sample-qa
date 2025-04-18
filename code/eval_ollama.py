from ollama import chat
from ollama import ChatResponse

from data_loader import get_qas

def main():
    qas = get_qas()

    for i, qa in enumerate(qas):
        full_question = f"""
        Clinical Note: ```{qa["transcription"]}```

        Question: {qa["question"]}

        Choices:
        {"\n".join(qa["options"])}
        """
        # print(full_question)
        
        response: ChatResponse = chat(model='llama3.2:1b', messages=[
        {
            'role': 'system',
            'content': 'You are an expert in answering multiple choice clinical questions about a clinical note. Respond only with the choice letter (A, B, C, or D) and nothing else.',
        },
        {
            'role': 'user',
            'content': full_question,
        },
        ])
        print(response['message']['content'])
        print(f"CORRECT ANSWER: {qa["correct_answer"]}")
        print("=========================================================")

        if i == 4:
            return

if __name__ == "__main__":
    main()