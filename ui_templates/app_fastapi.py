# ui_templates/app_fastapi.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.video_workflow import EnterpriseVideoPipeline, QualityCheckReport

app = FastAPI(title="Enterprise AI Video Generation Skill API", version="1.0.0")
pipeline = EnterpriseVideoPipeline(output_root=str(ROOT_DIR / "outputs"), kb_path=str(ROOT_DIR / "data"))


class VideoRequest(BaseModel):
    topic: str = "PSU-800W 鈦金級伺服器電源產品發表"
    target_audience: str = "企業決策者與技術團隊"
    tone: str = "專業、具科技感、權威"
    target_duration_sec: int = 28
    voice_profile: str = "zh-TW-Corporate-Pro"
    project_id: str = "api_video_job"


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "provider": pipeline.llm.provider,
        "model": pipeline.llm.model,
        "skill": "enterprise-video-generator"
    }


@app.post("/api/video/generate")
def generate_enterprise_video(req: VideoRequest):
    try:
        res = pipeline.run_full_workflow(
            topic=req.topic,
            target_audience=req.target_audience,
            tone=req.tone,
            target_duration_sec=req.target_duration_sec,
            voice_profile=req.voice_profile,
            project_id=req.project_id
        )
        return {
            "project_id": res["project_id"],
            "script": res["script"].model_dump(),
            "voiceover": res["voiceover"].model_dump(),
            "video": res["video"].model_dump(),
            "qc_report": res["qc_report"].model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
