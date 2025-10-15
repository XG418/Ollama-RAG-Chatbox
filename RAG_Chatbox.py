import streamlit as st
import pandas as pd
import ollama


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pypdf import PdfReader  


st.title("Jennie RAG ChatBox")




uploaded_file = st.file_uploader("Please upload file (CSV, Excel, or PDF)", type=["csv", "xlsx", "pdf"])
question = st.text_input("Please type in your question:")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
   
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((text[start:end], f"{start}-{end}"))
        start += step
    return chunks


if uploaded_file and question:
    try:
        docs = []  


        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
           
            df = pd.read_csv(uploaded_file, on_bad_lines="skip", encoding="utf-8")
            if df.empty:
                st.warning("The uploaded CSV has no rows.")
                st.stop()


           
            chunk_size_rows = 50
            for start in range(0, len(df), chunk_size_rows):
                chunk_df = df.iloc[start:start + chunk_size_rows]
               
                chunk_text = chunk_df.to_string(index=False)
                docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={"source": uploaded_file.name, "rows": f"{start}-{min(start + chunk_size_rows, len(df))}"}
                    )
                )


        elif name.endswith(".xlsx"):
            # Excel: automatically only read the first sheet
            xls = pd.ExcelFile(uploaded_file)
            sheet = xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
            if df.empty:
                st.warning("The selected sheet has no rows.")
                st.stop()


            chunk_size_rows = 50
            for start in range(0, len(df), chunk_size_rows):
                chunk_df = df.iloc[start:start + chunk_size_rows]
                chunk_text = chunk_df.to_string(index=False)
                docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={"source": f"{uploaded_file.name}:{sheet}", "rows": f"{start}-{min(start + chunk_size_rows, len(df))}"}
                    )
                )


        elif name.endswith(".pdf"):
           
            reader = PdfReader(uploaded_file)
            pages = []
            for i, page in enumerate(reader.pages):
                t = page.extract_text() or ""
                t = " ".join(t.split())  
                if t.strip():
                    pages.append(f"[Page {i+1}] {t}")


            if not pages:
                st.warning("No extractable text found in this PDF. If it is scanned, you may need OCR.")
                st.stop()


            full_text = "\n\n".join(pages)
            for txt, span in chunk_text(full_text, chunk_size=1200, overlap=200):
                docs.append(
                    Document(
                        page_content=txt,
                        metadata={"source": uploaded_file.name, "chars": span}
                    )
                )
        else:
            st.error("Unsupported file type.")
            st.stop()


 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embedding=embeddings)  


   
        results = db.similarity_search_with_score(question, k=3)
        if not results:
            st.warning("No relevant content found. Try another question.")
        else:
           
            context_text = "\n\n---\n\n".join(doc.page_content for doc, _score in results)


           
            prompt = (
                f"Please answer based ONLY on the context below:\n{context_text}\n\n"
                f"---\nQuestion: {question}\n"
                "If the context is insufficient, say 'Insufficient context'."
            )


            with st.spinner("Generating answer..."):
                resp = ollama.chat(
                    model="qwen2",  
                    messages=[{"role": "user", "content": prompt}]
                )


            st.write("🤖 Answer:", resp["message"]["content"])
            st.write("References:", [doc.metadata for doc, _score in results])


            with st.expander("Show retrieved context"):
                st.write(context_text)


    except Exception as e:
        st.error(f"Error processing file: {e}")



