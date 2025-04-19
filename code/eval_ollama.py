from ollama import chat
from ollama import ChatResponse

from data_loader import get_qas

def main():
    qas = get_qas()
    answers = []
    num_correct = 0
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
        answer = final_answer['message']['content']
        answers.append(answer)

        if answer == qa["correct_answer"]:
            num_correct += 1
        accuracy = num_correct/(i+1)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} questions. Accuracy={accuracy:.2f}")
        
        # print(f"CORRECT ANSWER: {qa["correct_answer"]}")
        # print("=========================================================")
    accuracy = num_correct/len(qas)
    print(accuracy)
    with open("answers_llama3_2_1b.txt", "w") as f:
        f.write("\n".join(answers))

if __name__ == "__main__":
    main()