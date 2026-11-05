# ui_templates/app_streamlit.py
import sys
from pathlib import Path

# 確保能 import 根目錄的 core 模組
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from core.video_workflow import EnterpriseVideoPipeline

st.set_page_config(page_title="企業 AI 影音生成戰情室", layout="wide")
st.title("🎬 企業級 AI 影音生成工作流程系統 (Enterprise Video Generator Skill)")
st.caption("基於 `ai-engineering-toolkit` 模組化架構：Step 1 腳本生成 ➔ Step 2 配音合成 ➔ Step 3 影片渲染 ➔ Step 4 輸出品質檢查")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = EnterpriseVideoPipeline(output_root=str(ROOT_DIR / "outputs"), kb_path=str(ROOT_DIR / "data"))

with st.sidebar:
    st.header("⚙️ 企業影片生成參數配置")
    topic = st.text_input("影片主題 (Topic)", value="PSU-800W 鈦金級伺服器電源產品發表")
    audience = st.text_input("目標受眾 (Target Audience)", value="雲端資料中心架構師與 IT 決策者")
    tone = st.selectbox("語氣與風格 (Tone)", ["專業、具科技感、權威", "嚴謹、清晰、具行動力", "溫暖、開放透明、充滿活力", "宏觀、誠懇、具社會責任感"])
    duration = st.slider("預計總時長 (秒)", min_value=15, max_value=60, value=28, step=4)
    voice = st.selectbox("配音聲線 (Voice Profile)", ["zh-TW-Corporate-Pro", "zh-TW-Corporate-Female"])
    project_id = st.text_input("專案代號 (Project ID)", value="streamlit_demo")
    run_btn = st.button("🚀 啟動四步影音生成與品質檢查", type="primary")

if run_btn:
    with st.spinner("正在依序執行：Step 1 腳本 ➔ Step 2 配音 ➔ Step 3 影片渲染 ➔ Step 4 品質檢核..."):
        res = st.session_state.pipeline.run_full_workflow(
            topic=topic,
            target_audience=audience,
            tone=tone,
            target_duration_sec=duration,
            voice_profile=voice,
            project_id=project_id
        )
        st.session_state.last_result = res

if "last_result" in st.session_state:
    res = st.session_state.last_result
    script = res["script"]
    vo = res["voiceover"]
    vid = res["video"]
    qc = res["qc_report"]

    t1, t2, t3, t4 = st.tabs([
        "📝 Step 1: 影片腳本 (Script)",
        "🎙️ Step 2: 配音與字幕 (Voiceover)",
        "🎥 Step 3: 影片與分鏡預覽 (Video)",
        "🛡️ Step 4: 輸出品質檢查 (QC Report)"
    ])

    with t1:
        st.subheader(f"📌 {script.video_title}")
        st.write(f"**受眾**: {script.target_audience} | **風格**: {script.tone} | **總時長**: `{script.total_duration_sec}s`")
        st.table([s.model_dump() for s in script.scenes])
        st.info(f"🎯 **片尾行動呼籲 (CTA)**: {script.call_to_action}")

    with t2:
        st.subheader("🎙️ 全片合成配音與 SRT 字幕")
        st.audio(vo.full_audio_path)
        st.write(f"**配音總時長**: `{vo.total_audio_duration_sec}s` | **聲線**: `{vo.voice_profile}`")
        st.table([a.model_dump() for a in vo.scene_audios])
        st.code(Path(vo.srt_subtitle_path).read_text(encoding="utf-8"), language="text")

    with t3:
        st.subheader("🎥 分鏡關鍵影格與影片產出")
        st.write(f"**AVI 影片路徑**: `{vid.video_path}` (`{vid.file_size_bytes:,} bytes`, `{vid.total_frames} frames`)")
        cols = st.columns(len(vid.scene_assets))
        for idx, asset in enumerate(vid.scene_assets):
            with cols[idx]:
                st.image(asset.keyframe_path, caption=f"Scene {asset.scene_id:02d} ({asset.duration_sec}s)")

    with t4:
        st.subheader("🛡️ 五維度自動化輸出品質檢查報告")
        if qc.overall_passed:
            st.success(qc.summary)
        else:
            st.error(qc.summary)
        st.table([r.model_dump() for r in qc.rules])
