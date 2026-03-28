import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import gc
import time
import os

def prepare_knowledge_base():
    start_time = time.time()
    print("Reading medic.csv (256k rows)...")
    try:
        medic_df = pd.read_csv('medic.csv', usecols=[0, 1, 2], low_memory=False)
        medic_df.columns = ['Description', 'Patient', 'Doctor']
    except Exception as e:
        print(f"Error loading medic.csv: {e}")
        return
    print("Reading convo.csv...")
    try:
        convo_df = pd.read_csv('convo.csv')
        convo_df.columns = ['slno', 'question', 'answer']
    except Exception as e:
        print(f"Error loading convo.csv: {e}")
        return
    all_docs = []
    for row in tqdm(medic_df.itertuples(index=False), total=len(medic_df), desc="Formatting Medical Data"):
        desc, pat, doc = map(lambda x: str(x).strip() if pd.notna(x) else "", row)
        content = f"Context: {desc}\nPatient: {pat}\nDoctor: {doc}"
        all_docs.append(Document(page_content=content, metadata={"source": "medical"}))
    
    del medic_df
    gc.collect()

    for row in tqdm(convo_df.itertuples(index=False), total=len(convo_df), desc="Formatting Conversations"):
        _, ques, ans = map(lambda x: str(x).strip() if pd.notna(x) else "", row)
        content = f"Question: {ques}\nAnswer: {ans}"
        all_docs.append(Document(page_content=content, metadata={"source": "convo"}))

    del convo_df
    gc.collect()

    print("\nInitializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    batch_size = 5000  
    vector_db = None

    print(f"Generating Embeddings for {len(all_docs)} documents in batches of {batch_size}...")
    
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        
        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            vector_db.add_documents(batch)
        
        del batch
        gc.collect()
        
        print(f"--- Indexed {min(i + batch_size, len(all_docs))} / {len(all_docs)} documents ---")

    print("\nSaving knowledge base to disk...")
    vector_db.save_local("medical_vector_db")
    
    total_time = (time.time() - start_time) / 60
    print(f"\nSUCCESS! Knowledge base saved to 'medical_vector_db'.")
    print(f"Total Time Elapsed: {total_time:.2f} minutes.")

if __name__ == "__main__":
    prepare_knowledge_base()
