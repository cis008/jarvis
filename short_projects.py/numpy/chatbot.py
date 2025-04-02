import openai
def chat_with_openai(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
    )
    return response['choices'][0]['message']['content']
def start_chatbot():
    print("👋 Welcome! I'm your chatbot. Type 'exit' to end the chat.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye! 👋")
            break
        response = chat_with_openai(user_input)
        print(f"Bot: {response}\n")
if __name__ == "__main__":
    start_chatbot()
    


