# core/agent_tools.py
import inspect
import json
from typing import Callable, Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.llm_engine import LLMEngine


class AgentDecision(BaseModel):
    thought: str = Field(
        description="目前的推理思考過程：分析還缺什麼資訊、為什麼要呼叫該工具或為什麼已經可以回答"
    )
    action: str = Field(
        description="要呼叫的工具名稱；若已經收集足夠資訊可直接回答，請填 'FINAL_ANSWER'"
    )
    # 關鍵修正：使用 JSON 字串取代 Dict[str, Any]，完美避開 Gemini/OpenAI 對 additionalProperties 的限制！
    action_input_json: str = Field(
        default="{}",
        description="傳入工具的參數，必須是合法的 JSON 字串 (例如 '{\"part_id\": \"PSU-800W\"}')；若 action 為 FINAL_ANSWER 則填 '{}'"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="當 action 為 FINAL_ANSWER 時，提供給使用者的最終完整回答"
    )


class ToolAgent:
    def __init__(self, llm_engine: Optional[LLMEngine] = None, max_steps: int = 5):
        self.llm = llm_engine or LLMEngine()
        self.max_steps = max_steps
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: List[str] = []

    # ==========================================
    # 註冊任何 Python 函數成為 Agent 的工具
    # ==========================================
    def register_tool(self, func: Callable):
        name = func.__name__
        sig = str(inspect.signature(func))
        doc = inspect.getdoc(func) or "無說明"
        self.tools[name] = func
        self.tool_descriptions.append(f"- 工具名稱: `{name}{sig}`\n  功能說明: {doc}")
        return func

    # ==========================================
    # 執行 ReAct 思考與工具呼叫迴圈
    # ==========================================
    def run(self, user_query: str, verbose: bool = True) -> Dict[str, Any]:
        tools_spec = "\n".join(self.tool_descriptions) if self.tool_descriptions else "無可用工具"
        
        system_prompt = (
            "你是一個具備自主推理與工具調用能力的 AI Agent。\n"
            "你可以使用以下工具來獲取資訊或執行計算：\n"
            f"{tools_spec}\n\n"
            "【執行規則】：\n"
            "1. 每次請先在 `thought` 中寫下你的分析。\n"
            "2. 若需要查詢資料或計算，請將 `action` 設為對應的工具名稱，並在 `action_input_json` 填入合法的 JSON 字串參數。\n"
            "3. 當你已經從觀察結果 (Observation) 獲得足夠資訊時，請將 `action` 設為 'FINAL_ANSWER'，並在 `final_answer` 寫出最終答案。"
        )

        scratchpad = f"【使用者任務】：{user_query}\n"
        trace_history = []

        for step in range(1, self.max_steps + 1):
            decision: AgentDecision = self.llm.generate_structured(
                prompt=scratchpad + f"\n請決定第 {step} 步的行動：",
                response_schema=AgentDecision,
                system_instruction=system_prompt
            )

            # 解析 JSON 字串為 Python Dict
            try:
                action_args = json.loads(decision.action_input_json) if decision.action_input_json.strip() else {}
                if not isinstance(action_args, dict):
                    action_args = {}
            except Exception as e:
                action_args = {}
                if verbose:
                    print(f"⚠️ [參數 JSON 解析警告]: {e} | 原始字串: {decision.action_input_json}")

            if verbose:
                print(f"🤖 [Step {step} 思考]: {decision.thought}")
                print(f"🛠️ [Step {step} 行動]: {decision.action} | 參數: {action_args}")

            # 判斷是否已經得出最終答案
            if decision.action == "FINAL_ANSWER" or decision.final_answer:
                ans = decision.final_answer or decision.thought
                if verbose:
                    print(f"✅ [最終回答]: {ans}")
                return {"answer": ans, "trace": trace_history, "steps": step}

            # 執行對應的 Python 工具並包含完整的 Error Handling
            tool_name = decision.action
            if tool_name not in self.tools:
                observation = f"錯誤：找不到名為 '{tool_name}' 的工具。請從可用工具清單中選擇。"
            else:
                try:
                    tool_func = self.tools[tool_name]
                    result = tool_func(**action_args)
                    observation = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                except Exception as e:
                    observation = f"工具執行發生異常 ({type(e).__name__}): {e}。請檢查傳入參數名稱與型別是否正確。"

            if verbose:
                print(f"👀 [Step {step} 觀察結果]: {observation}\n")

            trace_history.append({
                "step": step,
                "thought": decision.thought,
                "action": tool_name,
                "input": action_args,
                "observation": observation
            })

            scratchpad += (
                f"\n[Step {step}]\n"
                f"Thought: {decision.thought}\n"
                f"Action: {tool_name}({json.dumps(action_args, ensure_ascii=False)})\n"
                f"Observation: {observation}\n"
            )

        # 若超過最大步數仍未結束，強制總結
        fallback = self.llm.generate_text(
            prompt=scratchpad + "\n已達最大工具呼叫次數，請根據目前收集到的資訊直接回答使用者問題。"
        )
        return {"answer": fallback, "trace": trace_history, "steps": self.max_steps}


# ==========================================
# 🧪 點火測試 (執行 python -m core.agent_tools 驗證)
# ==========================================
if __name__ == "__main__":
    agent = ToolAgent()

    @agent.register_tool
    def check_inventory(part_id: str) -> dict:
        """查詢指定硬體料號 (例如 PSU-800W, FAN-120-B) 的目前庫存數量與單價(USD)"""
        mock_db = {
            "PSU-800W": {"stock": 45, "unit_price_usd": 120.5, "warehouse": "內湖一倉"},
            "FAN-120-B": {"stock": 120, "unit_price_usd": 15.0, "warehouse": "高雄二倉"},
        }
        return mock_db.get(part_id.upper(), {"error": f"查無料號 {part_id}"})

    @agent.register_tool
    def calculate_order_cost(unit_price: float, quantity: int, discount_rate: float = 1.0) -> float:
        """計算採購總金額：傳入單價 (unit_price)、數量 (quantity) 與折扣率 (discount_rate, 如 0.9 代表九折)"""
        return round(unit_price * quantity * discount_rate, 2)

    print("🚀 啟動 ReAct Agent 測試...\n")
    res = agent.run("客戶想採購 30 顆 PSU-800W，請先幫我確認庫存夠不夠？如果夠的話，給他打 9 折後總共是多少美金？")