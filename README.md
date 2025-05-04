# Chat Bot Furia

Autores:

- Henrique Yuji Zoppello Kanashiro (<henrique.zoppello@gmail.com>)

## **Aviso**

A leitura desta documentação e o entendimento da arquitetura deste projeto serão mais fáceis e melhor aproveitadas se você estiver familiarizado com os seguintes serviços: [LangChain](https://python.langchain.com/docs/introduction/) e [Google AI](https://python.langchain.com/docs/integrations/llms/google_ai/).

## 1. Contexto

Projeto desnvolvido para a etapa do desafio tecnico(Chat Bot)

O projeto consiste em um bot assitente que conversará com o usuario e ajudará a fornecer informações sobre a Furia.

O projeto adota como premissa a utilização de informações e noticias, conseguidas através de sites ou fornecidas pela própria Furia.

### 1.1 Objetivo

Desenvolver um chat bot usando o metodo de RAG(Retrival Augmented Generation) para alimentar o bot com informações.

## 2. Arquitetura

A arquitetura do Bot foi desenvolvida utilizando os serviços do LangChain.

Abaixo, a visão geral da arquitetura:

||
|:-:|
|<img src="fluxo_codigo.jpeg" width=60%>|

A arquitetura final da solução pode ser descrita da seguinte forma:

1. A partir de um input do usuario é iniciado uma chain que reformula o input com informações adicionais

2. Então outra chain é iniciada para passar o contexto do main prompt e os documentos

3. Por fim a rag_chain junta essas duas para gerar um resultado

### 2.1 Etapas do Processo

1. Armazenamento de informações:
As informações coletadas são separadas em documentos.md para serem vetorizadas.

2. embedding da informações:  
O modelo escolhido executa o embedding e coloca todos os documentos vetorizados em um banco de dados (chroma_db).

3. Alimentação do modelo com os documentos (RAG):
Nesse processo o LLM é alimentado com as informações coletadas

4. Retrieve das infomações:
O mesmo modelo usado para o embedding faz o retrieve das informações

5. Resposta do Bot:
A resposta é formulada levando em conta o historico do chat, os documentos recuperados e o prompt dado.

## 3. Serviços Utilizados

### 3.1 Google Gemini

O Google Gemini (GCS) foi utilizado como llm e como modelo de embedding

O gemini("gemini-2.0-flash") foi escolhido para ser o llm desse projeto por ser uma AI boa, eficiente e sem custo para uso.

### LangChain

LangChain é um framework para desenvolvimento de aplicações que usam large language models (LLMs)

Nesse projeto foram usadas todas essas bibliotecas:

- langchain.chains -> create_history_aware_retriever, create_retrieval_chain
- langchain.chains.combine_documents -> create_stuff_documents_chain
- langchain_community.vectorstores -> Chroma
- langchain_core.messages -> HumanMessage, AIMessage
- langchain_core.prompts -> ChatPromptTemplate, MessagesPlaceholder
- langchain_google_genai -> ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

## 4. Código do Projeto

O código-fonte deste projeto foi pensado de modo a agilizar a produção mas tentando manter a qualidade, ele esta separado por processos: Criação do Banco de dados, a divisão das chunks, o Embedding das informações e o retrieve dessas informações . Juntos, esses processos fazem o RAG que é necessario para alimentar as informções ao LLM.

### 4.1 Criação do banco de dados

Nesta etapa nos usamos o chroma db para ser o nosso banco de dados e o criamos apartir de codigos:

Exemplo de uso:

```python

db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db")


print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")
```

### 4.2  Divisão das chunks

Aqui fazemos as chunks que futuramente soferão o embedding para serem postas nos bancos de dados

Exemplo de uso:

```python
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
```

### 4.3 Embedding das Informações

O embedding das Informações é feito com o gimini modelo embedding-001 e passado para o banco de dados, esse embedding é feito no arquivo docs.py.

```python
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
```

### 4.3 Retrieve das Informações

O retrive é feito consultando o vector store e retornando as informações com criterios de Similaridade.

```python
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
```
