#%% Import Library ----------------------------------------------------------------------------------------------------
import sys
import os
import re
import json
import torch
import pypdf
import requests
import sqlite3
import hashlib
import chromadb
from pathlib import Path
from collections import defaultdict
from transformers import pipeline
from typing import Dict, List, Any, Generator, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup

# %% Define Classes ---------------------------------------------------------------------------------------------------
@dataclass
class NormalizedChunk:
    id_hash: str
    source_file: str
    section_heading: str
    page_number: int
    chunk_index: int
    content: str

#%% Build DataBase ------------------------------------------------------------------------------------------------------
class DocumentDatabase:
    """Manages raw file persistence and metadata using SQLite."""
    
    def __init__(self, sql_path: str = "doc_store.db", chroma_path: str = "chroma_db"):
        self.data_path = self.get_data_path()
        self.sql_path = str(self.data_path / sql_path)
        self.chroma_path = chroma_path
        self.collection = None
        self._init_db()

    def get_data_path(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        DB_PATH = project_root / "data"
        
        # Make "data" Folder
        DB_PATH.mkdir(parents=True, exist_ok=True)

        # Return The Root
        return DB_PATH

    def _init_db(self):
        # Initialize Vector DataBase
        client = chromadb.PersistentClient(path=self.data_path)
        self.collection = client.get_or_create_collection(self.chroma_path)

        # Initialize SQLite DataBase
        with sqlite3.connect(self.sql_path) as conn:
            # Create a standard table with primary key
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id_hash TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    section_heading TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            
            # Create a full-text search table
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5( 
                    id_hash,
                    content,
                );
            """)
            conn.commit()

            # Sync Inserts
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(id_hash, content) 
                    VALUES (new.id_hash, new.content);
                END;
            """)
            
            # Sync Deletes
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, id_hash, content) 
                    VALUES('delete', old.id_hash, old.content);
                END;
            """)

            # Sync Updates
            conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, id_hash, content) 
                VALUES('delete', old.id_hash, old.content);
                INSERT INTO documents_fts(id_hash, content) 
                VALUES (new.id_hash, new.content);
            END;
            """)
            conn.commit()

    def store_document(self, chunks: List[NormalizedChunk]) -> None:
        """
        Each input chunk dict requires:
        - 'id_hash': str
        - 'source_file': str
        - 'section_heading': str
        - 'page-number': int
        - 'chunk_index': int
        - 'content': str
        """

        # Store chunks in Chroma DataBase
        ids = [c.id_hash for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "section_heading": c.section_heading,
                "page-number": c.page_number,
                "chunk_index": c.chunk_index,
                "character_count": len(c.content),
            }
            for c in chunks
        ]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

        # Store chunks in SQLite DataBase
        with sqlite3.connect(self.sql_path) as conn:
            for c in chunks: 
                conn.execute("""
                    INSERT OR IGNORE INTO documents (id_hash, source_file, section_heading, page_number, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (c.id_hash, c.source_file, c.section_heading, c.page_number, c.chunk_index, c.content))
            conn.commit()

    def get_document(self, file_hash: str) -> Dict[str, Any]:
        with sqlite3.connect(self.sql_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM documents WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Document hash {file_hash} not found.")
            return {"filename": row[0], "file_type": row[1], "raw_bytes": row[2]}

    def retrieval(self, query: str, top_n: int = 50, k: int = 60, second_n: int = 20):
        # Get ranked results using BM25
        with sqlite3.connect(self.sql_path) as conn:
            sql_query = query.replace(" ", " OR ")
            cur = conn.cursor()
            sql = """
                SELECT id_hash, bm25(documents_fts, 0.0, 1.0) AS score
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?;
            """
            cur.execute(sql, (sql_query, top_n))
            results = cur.fetchall()
            bm25_results = [id_hash for id_hash, score in results]
            print(len(bm25_results))
            
        # Get ranked results using Vector Search
        results = self.collection.query(query_texts=[query], n_results=top_n)
        lookup_dict = dict(zip(results['ids'][0], results['documents'][0]))
        vector_results = results['ids'][0]

        rrf_scores = defaultdict(float)

        # Accumulate RRF score from BM25 ranks
        for rank, doc_id in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

        # Accumulate RRF score from Vector Search ranks
        for rank, doc_id in enumerate(vector_results, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

        # Sort documents by descending RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_docs = sorted_docs[:second_n]

        # Return the content
        results = [lookup_dict[i] for i, d in sorted_docs]
        return results

#%% Build File Loader
class MultiFormatLoader:
    """Extracts, normalizes, and chunks multi-format documents into plaintext."""

    def process_file(self, file_path: str) -> Generator[NormalizedChunk, None, None]:
        """Loads and processes a local file path."""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext in [".txt", ".text"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                yield from self._parse_text(f.read(), filename)
        elif ext in [".md", ".markdown"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                yield from self._parse_markdown(f.read(), filename)
        elif ext in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                yield from self._parse_html(f.read(), filename)
        elif ext == ".pdf":
            yield from self._parse_pdf(file_path, filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _clean_text(self, text: str) -> str:
        """Normalizes whitespace and removes control characters."""
        text = re.sub(r"[\r\t]", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    def _parse_text(self, text: str, source: str) -> Generator[NormalizedChunk, None, None]:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        for p in paragraphs:
            cleaned = self._clean_text(p)
            if cleaned:
                yield NormalizedChunk(
                    id_hash=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                    source_file=source,
                    section_heading=None,
                    page_number=1,
                    chunk_index=0,
                    content=cleaned
                )

    def _parse_markdown(self, text: str, source: str) -> Generator[NormalizedChunk, None, None]:
        current_heading = None
        current_block = []

        for line in text.splitlines():
            line_str = line.strip()
            heading_match = re.match(r"^(#{1,6})\s+(.*)", line_str)
            
            if heading_match:
                if current_block:
                    content = self._clean_text(" ".join(current_block))
                    if content:
                        yield NormalizedChunk(
                            id_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            source_file=source,
                            section_heading=current_heading,
                            page_number=1,
                            chunk_index=0,
                            content=cleaned
                        )
                    current_block = []
                current_heading = heading_match.group(2).strip()
            elif line_str:
                current_block.append(line_str)

        if current_block:
            content = self._clean_text(" ".join(current_block))
            if content:
                yield NormalizedChunk(
                    id_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    source_file=source,
                    section_heading=current_heading,
                    page_number=1,
                    chunk_index=0,
                    content=content
                )

    def _parse_html(self, text: str, source: str) -> Generator[NormalizedChunk, None, None]:
        soup = BeautifulSoup(text, "html.parser")
        
        # Strip script/style metadata elements
        for element in soup(["script", "style", "head", "meta"]):
            element.decompose()

        current_heading = None
        
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
            tag_name = element.name.lower()
            text_content = self._clean_text(element.get_text())

            if not text_content:
                continue

            if tag_name.startswith("h"):
                i = -1
                current_heading = text_content
            elif tag_name == "p":
                i += 1
                yield NormalizedChunk(
                    id_hash=hashlib.sha256(text_content.encode("utf-8")).hexdigest(),
                    source_file=source,
                    section_heading=current_heading,
                    page_number=1,
                    chunk_index=i,
                    content=text_content
                )

    def _parse_pdf(self, file_path: str, source: str) -> Generator[NormalizedChunk, None, None]:
        reader = pypdf.PdfReader(file_path)
        
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            page_text = page.extract_text() or ""
            
            # Simple heuristic extraction: first capitalized short line is treatment heading
            lines = [l.strip() for l in page_text.splitlines() if l.strip()]
            current_heading = None
            
            if lines and len(lines[0]) < 60 and lines[0].isupper():
                current_heading = lines[0]

            cleaned = self._clean_text(page_text)
            if cleaned:
                yield NormalizedChunk(
                    id_hash=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                    source_file=source,
                    section_heading=current_heading,
                    page_number=page_num,
                    chunk_index=0,
                    content=cleaned
                )

#%% Web Crawler -------------------------------------------------------------------------------------------------------
url = "https://en.wikipedia.org/wiki/Pittsburgh"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
response = requests.get(url, headers=headers)
assert response.status_code == 200

#%% Store Data to Database --------------------------------------------------------------------------------------------
# Initialize
database = DocumentDatabase()
loader = MultiFormatLoader()

# Split file to chunks
chunks = list(loader._parse_html(response.text, "Pittsburgh - Wikipedia"))
database.store_document(chunks)