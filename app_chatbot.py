# Importações necessárias
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Pode adicionar mais loaders conforme seus arquivos
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import gradio as gr
import os

# --- Parte 1: Configuração Inicial e Carregamento/Indexação dos Documentos ---
# Esta parte só precisa ser executada uma vez ou quando seus documentos mudarem.

# Define o diretório onde o ChromaDB irá persistir os dados.
# Isso evita reprocessar os embeddings toda vez que você inicia o app.
PERSIST_DIRECTORY = "./chroma_db"
DOCUMENTS_PATH = "./docs/" # Crie uma pasta 'docs' e coloque seus PDFs/TXTs aqui

# Verifica se o diretório de persistência do ChromaDB já existe
# Se não existir, processa os documentos e salva os embeddings.
if not os.path.exists(PERSIST_DIRECTORY):
    print("Processando documentos e criando embeddings pela primeira vez...")
    documents = []
    for filename in os.listdir(DOCUMENTS_PATH):
        filepath = os.path.join(DOCUMENTS_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        # Adicione mais tipos de arquivos conforme necessário (ex: .docx, .csv)
        # from langchain_community.document_loaders import Docx2txtLoader
        # elif filename.endswith(".docx"):
        #     loader = Docx2txtLoader(filepath)
        else:
            print(f"Tipo de arquivo não suportado, pulando: {filename}")
            continue
        documents.extend(loader.load())

    # Inicializa o modelo de embeddings do Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Cria e persiste o VectorStore (banco de dados vetorial)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIRECTORY)
    print("Embeddings criados e salvos com sucesso!")
else:
    print("Carregando embeddings existentes do ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print("Embeddings carregados!")

# Inicializa o LLM do Ollama (o modelo que vai gerar as respostas)
llm = Ollama(model="qwen2.5:1.5b")

# Cria a cadeia de RAG (Retrieval-Augmented Generation)
# Esta cadeia fará a busca no vectorstore e passará o contexto para o LLM.
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # k=3 significa que ele buscará os 3 documentos mais relevantes
    return_source_documents=True # Opcional: retorna os pedaços de texto que foram usados como base
)

# --- Parte 2: Função para a Interface Gráfica (Gradio) ---

def responder_pergunta(pergunta):
    """
    Esta função será chamada pelo Gradio para processar a pergunta do usuário.
    """
    print(f"\nRecebendo pergunta: {pergunta}")
    try:
        # Invoca a cadeia de RAG para obter a resposta
        result = qa_chain.invoke({"query": pergunta})
        resposta = result["result"]
        fontes = "\n\n**Fontes:**\n"
        
        # Adiciona as fontes se a opção return_source_documents for True
        if "source_documents" in result and result["source_documents"]:
            for i, doc in enumerate(result["source_documents"]):
                # Assumindo que os documentos têm um atributo 'metadata' com 'source' ou similar
                source_info = doc.metadata.get("source", f"Documento desconhecido {i+1}")
                fontes += f"- {source_info}\n"
                # Opcional: Adicionar o conteúdo do trecho para depuração, se quiser
                # fontes += f"  Conteúdo: {doc.page_content[:200]}...\n"
        else:
            fontes += "- Nenhuma fonte específica encontrada (o modelo pode ter usado conhecimento geral)."

        print(f"Resposta gerada: {resposta[:100]}...") # Print para o terminal de console
        return resposta + fontes
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")
        return f"Desculpe, ocorreu um erro ao processar sua pergunta: {e}"

# --- Parte 3: Configuração e Lançamento da Interface Gradio ---

# Cria a interface Gradio
# gr.Interface(fn, inputs, outputs, title, description)
# fn: a função Python que o Gradio deve chamar quando o usuário interagir
# inputs: o tipo de input que o usuário dará (ex: "text" para caixa de texto)
# outputs: o tipo de output que a função retornará (ex: "text" para texto simples)
iface = gr.Interface(
    fn=responder_pergunta,
    inputs=gr.Textbox(lines=5, placeholder="Digite sua pergunta aqui sobre os documentos...", label="Sua Pergunta"),
    outputs=gr.Markdown(label="Resposta do Modelo"), # Use Markdown para formatação rica
    title="🤖 Seu Assistente de Conhecimento Baseado em Documentos (Ollama + LangChain)",
    description="Faça perguntas sobre os documentos carregados para obter respostas contextualizadas. Certifique-se de que seus arquivos estão na pasta 'docs/'."
)

# Lança a interface. Ela abrirá automaticamente no seu navegador.
print("\nIniciando a interface Gradio...")
print(f"Certifique-se de ter seus arquivos em: {os.path.abspath(DOCUMENTS_PATH)}")
iface.launch(share=False) # 'share=True' cria um link público temporário (útil para compartilhar), mas é opcional e pode expor seu modelo.