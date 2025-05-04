import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import asyncio

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Define o persistent_directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define o modelo de embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Carrega o vector store existente com a função de embedding
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Cria um retriever para consultar o vector store
# `search_type` especifica o tipo de busca
# `search_kwargs` contém argumentos adicionais para a busca
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Cria o modelo do gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# prompt de contextualização de perguntas
# Esse system prompt ajuda a IA a entender que deve reformular a pergunta, baseado no histórico de chat para torná-la uma pergunta standalone
contextualize_q_system_prompt = (
    "Dado um histórico de conversa e a última pergunta do usuário "
    "que pode referenciar o contexto do histórico da conversa, "
    "formule uma pergunta independente que possa ser compreendida "
    "sem o histórico da conversa. NÃO responda à pergunta, apenas "
    "reformule-a se necessário e, caso contrário, retorne-a como está."
)

# Cria um template de prompt para contextualizar perguntas
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Cria um retriever que leva em conta o histórico de chat
# Isto usa o LLM para ajudar a reformular a pergunta com base no histórico de chat
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt para responder perguntas
# Esse prompt ajuda a IA a entender que tipo de resposta deve dar 
# baseado no contexto recuperado e indica o que fazer se não souber a resposta
with open("src/prompts/main.txt", "r") as file:
    qa_system_prompt = file.read()

# Cria um template de prompt para responder perguntas
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Cria uma chain para combinar documentos para responder perguntas
# `create_stuff_documents_chain` alimenta o LLM com todo o contexto recuperado
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Cria uma chain de recuperação que combina o history-aware retriever e a chain de perguntas e respostas
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Guarda o histórico de chat dos usuários
user_histories = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text

    if query.lower() == "exit":
        await update.message.reply_text("Conversa encerrada.")
        os._exit(0)  # Encerra o chat

    # Inicializa histórico do usuário se não existir
    if user_id not in user_histories:
        user_histories[user_id] = []

    
    history = user_histories[user_id]
    # Processa a query do usuário através da chain
    result = rag_chain.invoke({"input": query, "chat_history": history})

    # resposta da IA
    await update.message.reply_text(result["answer"])

    # Atualiza o histórico
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=result["answer"]))

# Função principal para iniciar o chat 
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot iniciado.")
    app.run_polling()