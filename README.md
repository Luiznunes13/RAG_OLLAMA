# 🧠 RAG_OLLAMA

Projeto de teste de sistema RAG (Retrieval-Augmented Generation) com integração ao modelo Ollama.

## 🚀 Tecnologias Utilizadas
- Python 3.12
- LangChain
- ChromaDB
- Ollama
- OpenAI / LLM local
- VS Code
- GitHub

## 🛠️ Como rodar o projeto

```bash
git clone https://github.com/Luiznunes13/RAG_OLLAMA.git
cd RAG_OLLAMA
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
python app_chatbot_V2.py
📂 RAG_OLLAMA/
├── app_chatbot_V2.py        # Script principal
├── chatbot_core.py          # Núcleo do chatbot com lógica de RAG
├── chroma_db/               # Base vetorial
├── embedding_cache/         # Cache de embeddings
├── docs/                    # Documentação adicional
└── requirements.txt         # Dependências do projeto

📚 Créditos
Desenvolvido por Luiz Nunes
