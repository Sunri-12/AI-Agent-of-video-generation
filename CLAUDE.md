# Project Context & Claude Code Guide (AI_Agent_System)

本專案整合 `ai-engineering-toolkit` 核心架構與 Claude Code Skill (`enterprise-video-generator`)，提供企業級 AI 影音生成四步工作流程。

## 🎯 已掛載之 Claude Code Skill
- **Skill 名稱**: `enterprise-video-generator`
- **Skill 路徑**: `.claude/skills/enterprise-video-generator/SKILL.md` (同步鏡像於 `.agents/skills/enterprise-video-generator/SKILL.md`)
- **CLI 執行器**: `.claude/skills/enterprise-video-generator/scripts/run_video_skill.py`

## 📦 核心模組列表 (`core/`)
1. `core.video_workflow.EnterpriseVideoPipeline`: 企業影片生成四步管線 (`step1_generate_script` -> `step2_generate_voiceover` -> `step3_generate_video` -> `step4_quality_check`)。
2. `core.llm_engine.LLMEngine`: 統一封裝 Gemini / OpenAI 結構化輸出與離線高擬真回退機制。
3. `core.data_loader.UniversalDataLoader`: 萬用異質文件解析器，負責載入 `data/enterprise_kb.md`。
4. `core.rag_pipeline.SimpleHybridRAG`: BM25 + Vector 混合檢索器。
5. `core.agent_tools.ToolAgent`: 跨廠牌自癒式 ReAct Agent，可透過 `create_video_skill_agent()` 自主調用影音生成工具。

## ⚡ 常用指令速查
- 執行 6 題完整測試套件：`python tests/run_test_suite.py`
- 執行單一專案全流程：`python .claude/skills/enterprise-video-generator/scripts/run_video_skill.py --mode full --topic "PSU-800W 鈦金級伺服器電源產品發表" --project-id "demo_psu800w"`
