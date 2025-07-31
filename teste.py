# Diagnóstico: quantos chunks foram indexados para 0083?
results = vectorstore.get(include=["metadatas"])["metadatas"]
chunks_0083 = [m for m in results if m.get("agency") == "0083"]
print("→ Chunks com agency=0083:", len(chunks_0083))

# Veja um exemplo de conteúdo bruto
if chunks_0083:
    # busque apenas texto, sem LLM
    docs = vectorstore.similarity_search(
        "técnico responsável",
        k=11,
        filter={"agency":"0083"}
    )
    for idx, d in enumerate(docs, 1):
        print(f"\n— Rank {idx} —\n{d.page_content[:200]}…\n")
else:
    print("Nenhum chunk com agency=0083! Verifique seu loader/split.")
