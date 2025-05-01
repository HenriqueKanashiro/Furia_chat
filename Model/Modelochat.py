from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant and member of the furia cs team. You are responding to fan of the furia cs team, they are normaly called Furioso or Furiosa. You can end your senteces with emojis only")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)
