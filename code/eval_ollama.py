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
        
        raw_response: ChatResponse = chat(model='llama3.2:1b', messages=[
        {
            'role': 'system',
            'content': 'You are an expert in answering multiple choice clinical questions about a clinical note.',
        },
        {
            'role': 'user',
            'content': full_question,
        },
        ])
        # log this somewhere?
        # print(response['message']['content'])
        
        final_answer: ChatResponse = chat(model='llama3.2:1b', messages=[
        {
            'role': 'system',
            'content': 'You are an expert in parsing out the answer given to a multiple choice question. Respond only with the capital letter of the answer that was chosen.',
        },
        {
            'role': 'user',
            'content': raw_response["message"]["content"],
        },
        ])
        print(final_answer['message']['content'])
        print(f"CORRECT ANSWER: {qa["correct_answer"]}")
        print("=========================================================")

        if i == 4:
            return

if __name__ == "__main__":
    main()