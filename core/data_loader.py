# core/data_loader.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import pdfplumber


@dataclass
class DocChunk:
    chunk_id: str
    content: str
    source: str  # 例如: "spec.pdf (Page 3)" 或 "logs.csv (Row 15)"
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalDataLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ==========================================
    # 萬用入口：丟檔案路徑或整個資料夾路徑都可以
    # ==========================================
    def load_any(self, path: str) -> List[DocChunk]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到路徑: {path}")

        if p.is_dir():
            all_chunks = []
            for file_path in sorted(p.rglob("*")):
                if file_path.is_file() and not file_path.name.startswith("."):
                    try:
                        all_chunks.extend(self._load_single_file(file_path))
                    except Exception as e:
                        print(f"⚠️ 略過無法解析的檔案 {file_path.name}: {e}")
            return all_chunks
        else:
            return self._load_single_file(p)

    def _load_single_file(self, file_path: Path) -> List[DocChunk]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in [".csv", ".xlsx", ".xls"]:
            return self.load_tabular(file_path)
        elif ext == ".json":
            return self.load_json(file_path)
        elif ext in [".md", ".txt", ".log"]:
            return self.load_text(file_path)
        else:
            raise ValueError(f"不支援的檔案格式: {ext}")

    # ==========================================
    # 1. PDF 讀取 (自動抽出文字 + 將表格轉成 Markdown!)
    # ==========================================
    def load_pdf(self, file_path: Path) -> List[DocChunk]:
        chunks = []
        filename = file_path.name
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                
                # 抽取表格並轉為 Markdown 格式，讓 LLM 看得懂結構
                tables = page.extract_tables()
                table_md_list = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    # 清理 None 與換行符號
                    cleaned = [[str(cell).replace("\n", " ").strip() if cell else "" for cell in row] for row in table]
                    header = "| " + " | ".join(cleaned[0]) + " |"
                    sep = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
                    body = "\n".join(["| " + " | ".join(r) + " |" for r in cleaned[1:]])
                    table_md_list.append(f"\n[表格資料]\n{header}\n{sep}\n{body}\n")

                full_page_text = text + "\n" + "\n".join(table_md_list)
                full_page_text = full_page_text.strip()
                if not full_page_text:
                    continue

                # 針對單頁內容做滑動視窗切塊 (保留跨段落語意)
                sub_texts = self._split_text(full_page_text)
                for sub_idx, sub_text in enumerate(sub_texts):
                    source_tag = f"{filename} (Page {page_idx})"
                    chunks.append(
                        DocChunk(
                            chunk_id=f"{filename}_p{page_idx}_c{sub_idx}",
                            content=sub_text,
                            source=source_tag,
                            metadata={"filename": filename, "page": page_idx}
                        )
                    )
        return chunks

    # ==========================================
    # 2. CSV / Excel 讀取 (轉成結構化 Key-Value 文本塊)
    # ==========================================
    def load_tabular(self, file_path: Path, rows_per_chunk: int = 5) -> List[DocChunk]:
        filename = file_path.name
        df = pd.read_csv(file_path) if file_path.suffix.lower() == ".csv" else pd.read_excel(file_path)
        df = df.dropna(how="all").fillna("N/A")

        chunks = []
        # 將每幾筆 Row 打包成一個 Chunk，避免單筆太碎或整張表塞爆 Token
        for start_idx in range(0, len(df), rows_per_chunk):
            sub_df = df.iloc[start_idx : start_idx + rows_per_chunk]
            lines = []
            for row_idx, row in sub_df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                lines.append(f"[Row {row_idx + 1}] {row_str}")
            
            end_idx = min(start_idx + rows_per_chunk, len(df))
            source_tag = f"{filename} (Rows {start_idx + 1}-{end_idx})"
            chunks.append(
                DocChunk(
                    chunk_id=f"{filename}_r{start_idx+1}_{end_idx}",
                    content="\n".join(lines),
                    source=source_tag,
                    metadata={"filename": filename, "start_row": start_idx + 1}
                )
            )
        return chunks

    # ==========================================
    # 3. JSON / 純文字讀取
    # ==========================================
    def load_json(self, file_path: Path) -> List[DocChunk]:
        filename = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                content = json.dumps(item, ensure_ascii=False, indent=2)
                chunks.append(
                    DocChunk(
                        chunk_id=f"{filename}_item_{idx}",
                        content=content,
                        source=f"{filename} (Item {idx})",
                        metadata={"filename": filename, "index": idx}
                    )
                )
        else:
            text = json.dumps(data, ensure_ascii=False, indent=2)
            for idx, sub_text in enumerate(self._split_text(text)):
                chunks.append(
                    DocChunk(
                        chunk_id=f"{filename}_c{idx}",
                        content=sub_text,
                        source=f"{filename} (Part {idx + 1})",
                        metadata={"filename": filename}
                    )
                )
        return chunks

    def load_text(self, file_path: Path) -> List[DocChunk]:
        filename = file_path.name
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = []
        for idx, sub_text in enumerate(self._split_text(text)):
            chunks.append(
                DocChunk(
                    chunk_id=f"{filename}_c{idx}",
                    content=sub_text,
                    source=f"{filename} (Section {idx + 1})",
                    metadata={"filename": filename}
                )
            )
        return chunks

    # ==========================================
    # 輔助：給 DataFrame 產生一段 LLM 專用的統計摘要 (資料分析題超好用)
    # ==========================================
    @staticmethod
    def get_dataframe_summary(file_path: str, sample_rows: int = 3) -> str:
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
        info_lines = [
            f"檔案名稱: {Path(file_path).name}",
            f"資料維度: 共 {len(df)} 筆資料, {len(df.columns)} 個欄位",
            f"欄位與型態: {dict(df.dtypes.astype(str))}",
            f"\n前 {sample_rows} 筆範例資料:\n{df.head(sample_rows).to_markdown(index=False)}"
        ]
        return "\n".join(info_lines)

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        results = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            results.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return results