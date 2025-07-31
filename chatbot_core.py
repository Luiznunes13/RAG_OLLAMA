import os
import shutil
from typing import List, Dict, Any, Optional
import io
import re # Adicionado para extrair metadados
import logging
import hashlib
import json

# --- Desabilitar Telemetria do ChromaDB ---
from chromadb.config import Settings
# A forma correta é instanciar as configurações e depois modificar o atributo
settings = Settings()
settings.anonymized_telemetry = False

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LangChain Imports ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,  # Fallback 1
    UnstructuredPDFLoader, # Fallback 2 (OCR)
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    TextLoader  # Loader específico para .txt
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_core.documents import Document # Adicionado para criação de Documento
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator


# --- Adicionando importações para OCR com Tesseract ---
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

class EmbeddingCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_checksum(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_embedding(self, text: str) -> Optional[List[float]]:
        checksum = self._get_checksum(text)
        cache_file = os.path.join(self.cache_dir, f"{checksum}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def set_embedding(self, text: str, embedding: List[float]):
        checksum = self._get_checksum(text)
        cache_file = os.path.join(self.cache_dir, f"{checksum}.json")
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)

# --- Tesseract Configuration ---
# Aponta para o executável do Tesseract. Verifique se o caminho está correto para sua instalação.
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    logger.warning("Tesseract não foi configurado ou não foi encontrado no caminho especificado. A extração OCR pode falhar.")

# --- Configuration ---
DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
EMBEDDING_CACHE_DIR = "./embedding_cache"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Global Variables ---
qa_chain: Optional[Any] = None
vectorstore: Optional[Chroma] = None

# --- Metadata Configuration for Self-Query Retriever ---
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="agency",
        description="O número da agência, composto por 4 dígitos. Exemplo: 0036, 0174, 0025.",
        type="string",
    ),
]
DOC_DESCRIPTION = "Documentos sobre o status de migração de agências. Cada documento está associado a um número de agência específico, que deve ser usado para filtrar as buscas."


# --- Document Loaders Configuration ---
# Mapping for non-PDF file types
LOADER_MAPPING = {
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
}

def extract_metadata(file_path: str) -> dict:
    """
    Extrai metadados (número da agência e fonte) do nome do arquivo.
    """
    filename = os.path.basename(file_path)
    agency_number = "unknown"
    
    # Tenta extrair o número da agência de 4 dígitos do início do nome do arquivo
    match = re.search(r'^(\d{4})', filename)
    if match:
        agency_number = match.group(1)
    
    metadata = {"source": file_path, "agency": agency_number}
    
    if agency_number != "unknown":
        logger.info("Metadados extraídos para %s: agência '%s'", filename, agency_number)
    else:
        logger.warning("Não foi possível extrair o número da agência de %s. Usando '%s'.", filename, agency_number)
        
    return metadata

def load_pdf_hybrid(file_path: str, metadata: dict) -> List[Document]:
    """
    Carrega um PDF usando uma estratégia de enriquecimento de dados:
    1. Extrai texto digital usando PyMuPDF.
    2. Extrai imagens de cada página e aplica OCR com Tesseract.
    3. Combina ambos os textos em um único Documento LangChain para o arquivo.
    """
    filename = os.path.basename(file_path)
    digital_text = ""
    ocr_text = ""
    
    logger.info("Iniciando extração híbrida para %s...", filename)

    # Etapa 1: Extração de Texto Digital com PyMuPDF
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        if docs:
            digital_text = "\n".join([doc.page_content for doc in docs if doc.page_content])
            logger.debug("Texto digital extraído de %s.", filename)
    except Exception as e:
        logger.warning("Falha ao extrair texto digital de %s: %s", filename, e)

    # Etapa 2: Extração de Texto de Imagens com OCR (Tesseract)
    try:
        pdf_doc = fitz.open(file_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                text = pytesseract.image_to_string(pil_image, lang='por')
                if text.strip():
                    ocr_text += text + "\n"
        
        if ocr_text:
            logger.debug("Texto de imagens (OCR) extraído de %s.", filename)
    except Exception as e:
        logger.warning("Falha ao extrair texto de imagens em %s: %s", filename, e)

    # Etapa 3: Combina os textos e cria um único Documento
    combined_text = digital_text
    if ocr_text:
        combined_text += "\n\n--- TEXTO EXTRAÍDO DE IMAGENS (OCR) ---\n\n" + ocr_text
    
    if not combined_text.strip():
        logger.error("Nenhuma informação pôde ser extraída de %s.", filename)
        return []

    # Usa os metadados pré-extraídos
    return [Document(page_content=combined_text, metadata=metadata)]


def load_documents(path: str) -> List[Any]:
    """
    Carrega todos os documentos suportados do diretório especificado.
    Usa a estratégia híbrida para PDFs.
    """
    all_documents = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # --- Ponto único de extração de metadados ---
            metadata = extract_metadata(file_path)
            
            docs_to_add = []
            if file_ext == ".pdf":
                # Passa os metadados para a função de carregamento
                docs_to_add = load_pdf_hybrid(file_path, metadata)
            elif file_ext in LOADER_MAPPING:
                loader_class = LOADER_MAPPING[file_ext]
                try:
                    logger.info("Carregando %s com %s.", os.path.basename(file_path), loader_class.__name__)
                    loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    # Adiciona os metadados extraídos a cada documento
                    for doc in loaded_docs:
                        doc.metadata.update(metadata)
                    docs_to_add = loaded_docs
                except Exception as e:
                    logger.error("Falha ao carregar %s: %s", file_path, e)
            
            all_documents.extend(docs_to_add)

    return all_documents

def get_vectorstore(force_reindex: bool = False) -> Chroma:
    """
    Initializes or loads the Chroma vector store.
    If `force_reindex` is True, it deletes the old database and creates a new one.
    """
    global vectorstore
    if vectorstore is not None and not force_reindex:
        return vectorstore

    if os.path.exists(DB_DIR) and force_reindex:
        logger.info("Removendo banco de dados vetorial existente em: %s", DB_DIR)
        shutil.rmtree(DB_DIR)
        # Limpa também o cache de embeddings se a reindexação for forçada
        if os.path.exists(EMBEDDING_CACHE_DIR):
            shutil.rmtree(EMBEDDING_CACHE_DIR)


    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embedding_cache = EmbeddingCache(EMBEDDING_CACHE_DIR)

    if os.path.exists(DB_DIR):
        logger.info("Carregando banco de dados vetorial existente de: %s", DB_DIR)
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return vectorstore

    logger.info("Nenhum banco de dados vetorial encontrado. Criando um novo em: %s", DB_DIR)
    logger.info("Carregando documentos de: %s", DOCS_DIR)
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        logger.info("Pasta '%s' criada. Adicione seus documentos aqui.", DOCS_DIR)
        return None

    documents = load_documents(DOCS_DIR)
    if not documents:
        logger.warning("Nenhum documento encontrado para indexar.")
        return None

    # Validação para garantir que todos os documentos tenham uma agência válida
    validated_documents = [
        doc for doc in documents if doc.metadata.get("agency") and doc.metadata["agency"] != "unknown"
    ]
    
    # Log para documentos filtrados
    filtered_count = len(documents) - len(validated_documents)
    if filtered_count > 0:
        logger.warning("%d documentos foram filtrados por não terem uma agência válida.", filtered_count)

    documents = validated_documents
    if not documents:
        logger.error("Nenhum documento com metadados de agência válidos foi encontrado para indexação.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ": ", " ", ""],
    )
    splits = text_splitter.split_documents(documents)

    logger.info("Criando e persistindo o banco de dados vetorial com %d trechos de texto...", len(splits))
    
    # Inicializa o ChromaDB vazio que será populado
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Processa os splits em lotes para adicionar ao Chroma com cache
    batch_size = 32 # Lotes de 32 documentos por vez
    for i in range(0, len(splits), batch_size):
        batch_splits = splits[i:i+batch_size]
        texts_to_embed = [doc.page_content for doc in batch_splits]
        
        # Verifica o cache
        cached_embeddings = [embedding_cache.get_embedding(text) for text in texts_to_embed]
        
        # Identifica quais textos realmente precisam de embedding
        texts_needing_embedding = [
            text for text, cached_emb in zip(texts_to_embed, cached_embeddings) if cached_emb is None
        ]
        
        # Calcula os novos embeddings se necessário
        if texts_needing_embedding:
            new_embeddings = embeddings.embed_documents(texts_needing_embedding)
            
            # Salva os novos embeddings no cache
            for text, new_emb in zip(texts_needing_embedding, new_embeddings):
                embedding_cache.set_embedding(text, new_emb)
            
            # Re-lê do cache para garantir a consistência
            cached_embeddings = [embedding_cache.get_embedding(text) for text in texts_to_embed]

        # Adiciona os documentos com seus embeddings ao ChromaDB
        # Filtra qualquer 'None' que possa ter restado, embora não deva acontecer
        valid_embeddings = [emb for emb in cached_embeddings if emb is not None]
        if len(valid_embeddings) == len(batch_splits):
            vectorstore.add_texts(
                texts=[doc.page_content for doc in batch_splits],
                metadatas=[doc.metadata for doc in batch_splits],
                embeddings=valid_embeddings
            )
        else:
            logger.error("Falha ao obter embeddings para um lote, pulando.")

    logger.info("Banco de dados vetorial criado com sucesso!")
    vectorstore.persist() # Garante que tudo seja salvo no disco
    
    # --- DEBUGGING: Verifique se há chunks indexados para agências específicas ---
    if vectorstore:
        agencies_to_debug = ["0083", "0023", "0036"]
        all_docs_content_for_debug = vectorstore.get(include=["documents", "metadatas"])
        all_meta_for_debug = all_docs_content_for_debug.get("metadatas", [])
        all_documents_for_debug = all_docs_content_for_debug.get("documents", [])

        for agency_to_debug in agencies_to_debug:
            logger.info(f"--- INICIANDO VERIFICAÇÃO DE DEBUG PARA AGÊNCIA {agency_to_debug} ---")
            try:
                # 1. Veja quantos chunks têm metadata para a agência
                chunks_for_agency = [m for m in all_meta_for_debug if m.get("agency") == agency_to_debug]
                logger.info(f"Encontrei {len(chunks_for_agency)} chunks para agência {agency_to_debug}")

                # 2. Inspecione um conteúdo
                if chunks_for_agency:
                    # Busque manualmente um trecho parecido
                    docs = vectorstore.similarity_search(
                        "NOME DO TÉCNICO RESPONSÁVEL",
                        k=11,
                        filter={"agency": agency_to_debug},
                    )
                    if docs:
                        for idx, d in enumerate(docs, 1):
                            logger.info(f"\n— Rank {idx} (Agência {agency_to_debug}) —\n{d.page_content[:200]}…\n")
                    else:
                        logger.info(f"A busca por similaridade para agência {agency_to_debug} não retornou resultados.")

                    # Fallback de text-match para depuração
                    logger.info(f"--- INICIANDO FALLBACK DE TEXT-MATCH PARA AGÊNCIA {agency_to_debug} ---")
                    found_in_fallback = False
                    for i, content in enumerate(all_documents_for_debug):
                        meta = all_meta_for_debug[i]
                        if meta.get("agency") == agency_to_debug and "Flávio da Silva Alves" in content:
                            logger.info(f"Achei 'Flávio da Silva Alves' no chunk (via text-match, agência {agency_to_debug}): {content}")
                            found_in_fallback = True
                    if not found_in_fallback:
                        logger.info(f"Texto 'Flávio da Silva Alves' não encontrado em nenhum chunk da agência {agency_to_debug} via text-match.")
                    logger.info(f"--- FIM DO FALLBACK DE TEXT-MATCH PARA AGÊNCIA {agency_to_debug} ---")

            except Exception as e:
                logger.error(f"Erro durante a verificação de debug para agência {agency_to_debug}: {e}")
            logger.info(f"--- FIM DA VERIFICAÇÃO DE DEBUG PARA AGÊNCIA {agency_to_debug} ---")

    return vectorstore

def initialize_chatbot() -> None:
    """
    Initializes the chatbot by setting up the QA chain.
    """
    global qa_chain, vectorstore

    vectorstore = get_vectorstore()

    if vectorstore is None:
        logger.warning("Vectorstore não pôde ser inicializado. O chatbot pode não funcionar.")
        return

    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=DOC_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        structured_query_translator=ChromaTranslator(),
        enable_limit=False,
        verbose=True,
        search_kwargs={"k": 11}
    )

    prompt_template = (
        "Você é um assistente que responde usando só o contexto.\n"
        "Se não souber, responda: “Não tenho informações…”\n\n"
        "Contexto:\n{context}\n\n"
        "Pergunta:\n{question}\n\n"
        "Resposta:"
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    document_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=document_chain,
        return_source_documents=True,
    )
    logger.info("QA chain inicializada com sucesso.")

def get_chatbot_response(query: str) -> str:
    """
    Gets a response from the chatbot.
    """
    global qa_chain
    if qa_chain is None:
        initialize_chatbot()
        if qa_chain is None:
            return "O chatbot não está inicializado. Verifique se há documentos na pasta 'docs' e reinicie."

    try:
        # A chain RetrievalQA espera um dicionário com a chave "query"
        result = qa_chain.invoke({"query": query})
        
        # Log dos documentos fonte
        source_docs = result.get("source_documents")
        if source_docs:
            logger.info("Documentos fonte utilizados para a resposta:")
            for doc in source_docs:
                source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                page_number = doc.metadata.get('page')

                # Constrói a mensagem de log
                if page_number is not None:
                    # Adiciona 1 porque as páginas são geralmente 0-indexadas
                    log_message = f"  - Arquivo: {source_file}, Página: {page_number + 1}"
                else:
                    log_message = f"  - Arquivo: {source_file} (sem informação de página)"
                
                logger.info(log_message)

        # A resposta está na chave "result"
        return result["result"]
    except Exception as e:
        return f"Ocorreu um erro ao processar sua pergunta: {e}"

def reindex_documents() -> str:
    """
    Forces re-indexing of all documents.
    """
    global qa_chain, vectorstore
    try:
        qa_chain = None
        vectorstore = None
        get_vectorstore(force_reindex=True)
        initialize_chatbot()
        return "Re-indexação concluída com sucesso. O chatbot está pronto."
    except Exception as e:
        return f"Falha na re-indexação: {e}"

def list_documents_in_folder() -> List[str]:
    """
    Lists files in the './docs' folder for display in the UI.
    """
    if not os.path.exists(DOCS_DIR):
        return ["A pasta 'docs' não foi encontrada."]
    try:
        files = os.listdir(DOCS_DIR)
        if not files:
            return ["Nenhum documento encontrado na pasta 'docs'."]
        return files
    except Exception as e:
        return [f"Erro ao listar documentos: {e}"]

def list_indexed_sources() -> List[str]:
    """
    Lists the source documents currently indexed in the vector store.
    """
    global vectorstore
    if vectorstore is None:
        vs = get_vectorstore()
        if vs is None:
            return ["O banco de dados vetorial ainda não foi criado."]
    else:
        vs = vectorstore

    try:
        results = vs.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return ["Nenhuma fonte indexada encontrada."]

        sources = sorted(list(set(meta['source'] for meta in results['metadatas'] if 'source' in meta)))
        return [os.path.basename(s) for s in sources]
    except Exception as e:
        return [f"Erro ao obter fontes indexadas: {e}"]
