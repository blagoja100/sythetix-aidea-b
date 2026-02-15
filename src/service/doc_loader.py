import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import Logger

logger = Logger("DocLoader")

class TxtFileLoader:
    def __init__(self, data_path: str = "data/txt/"):
        self.data_path = data_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

    def _read_file(self, file_path: str) -> str:       
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            logger.log(f"Warning: {os.path.basename(file_path)} is not UTF-8. Trying fallback encoding.")
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        return ""

    def load_txt_file_chunks(self) -> Dict[str, List[str]]:
        all_chunks = {}
        logger.log(f"Loading files from {self.data_path}")
        logger.log(f"Current directory: {os.getcwd()}")
        if os.path.exists(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith(".txt"):
                    logger.log(f"Processing {filename}")
                    file_path = os.path.join(self.data_path, filename)
                    
                    raw_text = self._read_file(file_path)
                    if not raw_text:
                        continue

                    chunks = self.splitter.split_text(raw_text)
                    all_chunks[filename] = chunks
                    logger.log(f"Processed {filename}: {len(chunks)} chunks")
        
        return all_chunks
