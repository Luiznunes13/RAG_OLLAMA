# app_chatbot.py
import gradio as gr
import chatbot_core

# --- Inicialização ---
# Inicializa o chatbot ao iniciar a aplicação
# A mensagem de status será impressa no console onde o script é executado
print("Inicializando o chatbot, por favor aguarde...")
qa_chain_instance = chatbot_core.initialize_chatbot()
print("Chatbot inicializado com sucesso!")

# --- Funções Wrapper para o Gradio ---
def gradio_response_wrapper(query, chat_history):
    """
    Função wrapper para o chat do Gradio. Recebe uma pergunta e o histórico
    do chat, obtém uma resposta do chatbot e retorna o histórico atualizado.
    """
    response = chatbot_core.get_chatbot_response(query)
    chat_history.append((query, response))
    return "", chat_history

def reindex_and_update_status():
    """Função para ser chamada pelo botão de reindexação."""
    status_message = chatbot_core.reindex_documents()
    # Atualiza a lista de documentos após a reindexação
    doc_list = chatbot_core.list_documents_in_folder()
    return status_message, doc_list

def get_initial_doc_list():
    """Pega a lista de documentos ao carregar a UI."""
    return chatbot_core.list_documents_in_folder()

# --- Construção da Interface Gradio ---
with gr.Blocks(theme=gr.themes.Soft(), title="Assistente de Conhecimento") as iface:
    gr.Markdown("# 🤖 Seu Assistente de Conhecimento Baseado em Documentos")
    gr.Markdown("Faça perguntas sobre os documentos carregados para obter respostas contextualizadas. Os documentos devem estar na pasta `docs/`.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                bubble_full_width=False,
                height=500
            )
            # O gr.State foi removido; o componente chatbot agora gerencia o histórico.

            question_box = gr.Textbox(
                lines=3,
                placeholder="Digite sua pergunta aqui...",
                label="Sua Pergunta",
                show_label=False
            )
            submit_btn = gr.Button("Enviar Pergunta", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Gestão da Base de Conhecimento")
            
            reindex_btn = gr.Button("Forçar Re-indexação dos Documentos")
            status_output = gr.Textbox(label="Status da Re-indexação", interactive=False)
            
            gr.Markdown("---") # Separador
            # Exibe a lista de documentos em caixa de texto com múltiplas linhas e scrollbar
            doc_list_output = gr.Textbox(
                label="Documentos na Base de Conhecimento",
                interactive=False,
                lines=10,
                placeholder="Nenhum documento listado ainda...",
            )
            
    # Ações da Interface
    iface.load(get_initial_doc_list, None, doc_list_output)
    submit_btn.click(
        fn=gradio_response_wrapper,
        inputs=[question_box, chatbot],
        outputs=[question_box, chatbot],
    )
    question_box.submit(
        fn=gradio_response_wrapper,
        inputs=[question_box, chatbot],
        outputs=[question_box, chatbot],
    )
    reindex_btn.click(
        fn=reindex_and_update_status,
        inputs=None,
        outputs=[status_output, doc_list_output],
    )

iface.launch(share=False) # 'share=True' cria um link público temporário