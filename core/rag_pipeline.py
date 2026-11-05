# core/rag_pipeline.py
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import chromadb
from rank_bm25 import BM25Okapi

from core.data_loader import DocChunk
from core.llm_engine import LLMEngine


# 定義強制附帶來源引用的結構化輸出格式
class RAGResponse(BaseModel):
    answer: str = Field(description="根據參考資料所生成的完整回答，若資料不足請明確說明無法從現有資料得知")
    citations: List[str] = Field(description="回答所引用的資料來源清單，例如 ['spec.pdf (Page 3)']")
    confidence: str = Field(description="回答信心度: High, Medium, Low")


class SimpleHybridRAG:
    def __init__(self, llm_engine: Optional[LLMEngine] = None):
        self.llm = llm_engine or LLMEngine()
        self.chunks: List[DocChunk] = []
        self.chunk_map: Dict[str, DocChunk] = {}
        self.bm25: Optional[BM25Okapi] = None

        # 使用純記憶體 ChromaDB，不需架設伺服器、每次重跑乾淨不殘留
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="f1_hybrid_kb")

    # 支援中文字元與英數型號(如 PSU-500W)的輕量級混合斷詞器
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        # 抓取連續英數字型號 (如 psu-500w, err_01) 或單一中文字元
        tokens = re.findall(r"[a-z0-9_\-]+|[\u4e00-\u9fff]", text)
        return tokens

    # ==========================================
    # 1. 灌入資料 (同時建立 Vector 索引與 BM25 索引)
    # ==========================================
    def ingest(self, chunks: List[DocChunk]):
        if not chunks:
            print("⚠️ 沒有可匯入的 DocChunk！")
            return

        self.chunks.extend(chunks)
        for c in chunks:
            self.chunk_map[c.chunk_id] = c

        # (A) 建立 BM25 關鍵字索引
        tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # (B) 建立 Chroma 向量索引 (分批寫入避免單次過大)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.content for c in batch],
                metadatas=[{"source": c.source, **c.metadata} for c in batch]
            )
        print(f"✅ 成功匯入 {len(chunks)} 個區塊！(目前知識庫總計: {len(self.chunks)} 塊)")

    # ==========================================
    # 2. RRF 混合檢索 (Vector + BM25 雙劍合璧)
    # ==========================================
    def search(self, query: str, top_k: int = 4) -> List[DocChunk]:
        if not self.chunks:
            return []

        k_candidates = min(top_k * 2, len(self.chunks))
        rrf_scores: Dict[str, float] = {}

        # (A) Vector Search 排名
        vec_res = self.collection.query(query_texts=[query], n_results=k_candidates)
        vec_ids = vec_res["ids"][0] if vec_res["ids"] else []
        for rank, cid in enumerate(vec_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        # (B) BM25 關鍵字排名 (精準命中料號、型號、專有名詞)
        if self.bm25:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            # 取出分數最高的前 k_candidates 名
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_candidates]
            for rank, idx in enumerate(top_bm25_indices):
                if bm25_scores[idx] > 0:
                    cid = self.chunks[idx].chunk_id
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)

        # 依照 RRF 綜合分數由高到低排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [self.chunk_map[cid] for cid in sorted_ids]

    # ==========================================
    # 3. 檢索增強問答 (自動帶出參考來源 Citation)
    # ==========================================
    def query(self, question: str, top_k: int = 4, structured: bool = True) -> RAGResponse | str:
        retrieved_chunks = self.search(question, top_k=top_k)
        
        context_str = "\n\n---\n\n".join(
            [f"【來源: {c.source}】\n{c.content}" for c in retrieved_chunks]
        )

        system_prompt = (
            "你是一個嚴謹的企業級 AI 知識庫助手。\n"
            "請『嚴格根據』以下提供的【參考資料】回答使用者的問題。\n"
            "規則：\n"
            "1. 絕對不可編造參考資料以外的數據或事實。\n"
            "2. 若參考資料不足以回答問題，請直接表明『根據現有資料無法確認』。\n"
            "3. 必須明確標註你引用了哪一個【來源】。"
        )

        user_prompt = f"【參考資料】：\n{context_str}\n\n【使用者問題】：\n{question}"

        if structured:
            return self.llm.generate_structured(
                prompt=user_prompt,
                response_schema=RAGResponse,
                system_instruction=system_prompt
            )
        else:
            return self.llm.generate_text(
                prompt=user_prompt,
                system_instruction=system_prompt
            )


# ==========================================
# 🧪 點火測試 (執行 python -m core.rag_pipeline 驗證)
# ==========================================
if __name__ == "__main__":
    # 模擬現場拿到的幾份混合規格書與異常紀錄
    mock_chunks = [
        DocChunk(
            chunk_id="doc_1",
            content="型號 PSU-800W 伺服器電源供應器，輸入電壓範圍為 100V-240V，最高轉換效率達 94% (80 Plus 鈦金級)，工作溫度上限為 55°C。",
            source="PSU_Spec_2026.pdf (Page 4)"
        ),
        DocChunk(
            chunk_id="doc_2",
            content="錯誤代碼 ERR-092 代表散熱風扇轉速低於 800 RPM。標準處置 SOP：請先檢查風扇排線 J4 是否鬆脫，若正常則更換風扇模組 FAN-120-B。",
            source="Maintenance_SOP.md (Section 2)"
        ),
        DocChunk(
            chunk_id="doc_3",
            content="型號 PSU-500W 標準型電源，最高轉換效率為 88%，工作溫度上限為 45°C。",
            source="PSU_Spec_2026.pdf (Page 2)"
        ),
    ]

    rag = SimpleHybridRAG()
    rag.ingest(mock_chunks)

    test_q = "請問 PSU-800W 的轉換效率跟工作溫度上限是多少？如果遇到 ERR-092 要換哪個料號的模組？"
    print(f"\n🔍 測試提問: {test_q}\n")

    res = rag.query(test_q, top_k=2, structured=True)
    print("✅ [Hybrid RAG 結構化問答結果]:")
    print(f"💡 回答: {res.answer}")
    print(f"📚 引用來源: {res.citations}")
    print(f"🎯 信心度: {res.confidence}")