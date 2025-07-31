import os
import pandas as pd
import warnings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import concurrent.futures

# Suppress openpyxl stylesheet warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

def read_file(file_path):
    """Read content from a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                return {"type": "text", "content": f.read(), "file": os.path.basename(file_path)}
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
            return {"type": "excel", "content": df, "file": os.path.basename(file_path)}
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return {"type": "excel", "content": df, "file": os.path.basename(file_path)}
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise Exception(f"Failed to read {file_path}: {str(e)}")

def row_to_text(row):
    """Convert a DataFrame row to a textual representation."""
    return ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])

def process_file(file_path, text_splitter):
    """Process a single file and return texts and metadata."""
    texts = []
    metadatas = []
    try:
        document = read_file(file_path)
        filename = document["file"]
        if document["type"] == "text":
            # Split text into chunks for TXT and MD files
            chunks = text_splitter.split_text(document["content"])
            for chunk_id, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({"file": filename, "chunk_id": chunk_id})
        elif document["type"] == "excel":
            # Convert each row to text for XLSX, XLS, and CSV files
            df = document["content"]
            for row_id, row in df.iterrows():
                text = row_to_text(row)
                if text.strip():  # Only add non-empty texts
                    texts.append(text)
                    metadatas.append({"file": filename, "row_id": row_id})
        return texts, metadatas
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return [], []

def ingest_folder_to_faiss(folder_path, openai_api_key, batch_size=100, max_workers=4):
    """Ingest all files in the folder into FAISS using OpenAI embeddings."""
    # Initialize OpenAI embeddings with the large model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large")

    # Initialize text splitter with larger chunk size for efficiency
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    # Collect all file paths
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

    # Process files in parallel
    texts = []
    metadatas = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path, text_splitter): file_path for file_path in file_paths}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(file_paths), desc="Processing files"):
            file_texts, file_metadatas = future.result()
            texts.extend(file_texts)
            metadatas.extend(file_metadatas)

    # Create FAISS index in batches
    if texts:
        print(f"Creating FAISS index with {len(texts)} texts...")
        vector_store = None
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding and indexing"):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            if i == 0:
                vector_store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
            else:
                vector_store.add_texts(batch_texts, metadatas=batch_metadatas)
        return vector_store
    else:
        raise ValueError("No valid data found to ingest into FAISS")

# Example usage
folder_path = "Knowledge_Base"  # Replace with your folder path
openai_api_key = ""  # Replace with your OpenAI API key
vector_store = ingest_folder_to_faiss(folder_path, openai_api_key)

# Save the FAISS index to disk (optional)
vector_store.save_local("faiss_index")

