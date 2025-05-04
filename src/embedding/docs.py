# Vetorizar documentos
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Define o diretorio que contém os arquivos de texto e o persistent_directory
current_dir = os.path.dirname(os.path.abspath(__file__))
info_dir = os.path.join(current_dir, "info")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db")

print(f"Infos directory: {info_dir}")
print(f"Persistent directory: {persistent_directory}")

# Checar se o Chroma vector store já existe
if not os.path.exists(persistent_directory):
    print("Persistent directory nao existe. Inicializando vector store...")

    # Confere se o info_dir existe
    if not os.path.exists(info_dir):
        raise FileNotFoundError(
            f"O Diretorio {info_dir} não existe. Reveja o path."
        )

    # Lista todos os arquivos de texto no diretório
    info_files = [f for f in os.listdir(info_dir) if f.endswith(".md")]

    # Lê o conteúdo de cada arquivo e armazena com a metadata
    documents = []
    for info_file in info_files:
        file_path = os.path.join(info_dir, info_file)
        loader = TextLoader(file_path, encoding="utf-8")
        info_docs = loader.load()
        for doc in info_docs:
            # Adiciona metadata a cada documento indicando a Source
            doc.metadata = {"source": info_file}
            documents.append(doc)

    # Separa os documentos em chunks
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Mostra informações sobre os documentos divididos splitados
    print("\n--- Informação das chunks ---")
    print(f"Numero de chunks : {len(docs)}")

    # Cria os embeddings
    print("\n--- Criando embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    print("\n--- embeddings criados ---")

    # Cria o banco vetorizado e salva de forma persistente
    print("\n--- Criando e persistindo o vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Vector store Criado e persistido ---")

else:
    print("Vector store ja existe. Sem necessidade de inicialização.")