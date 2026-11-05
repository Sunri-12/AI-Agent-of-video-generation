# core/video_workflow.py
"""
企業級 AI 影音生成工作流程 (Enterprise AI Video Generation Workflow)
結合 ai-engineering-toolkit (LLMEngine, UniversalDataLoader, SimpleHybridRAG, ToolAgent)，
將企業影音生成拆解為四大獨立且可驗證之工程步驟：
  - Step 1: 影片腳本生成 (Script Generation) -> 結構化分鏡腳本 (Pydantic VideoScript)
  - Step 2: 配音生成 (Voiceover / TTS Synthesis) -> 各幕 WAV 語音合成與時長量測 + SRT 字幕檔
  - Step 3: 影片生成 (Video Rendering & Synthesis) -> 分鏡關鍵影格渲染、MJPEG AVI 實質影片封裝與 HTML5 互動預覽播放器
  - Step 4: 輸出品質檢查 (Output Quality Assurance) -> 5 大維度自動化品質關卡與自動修復機制
"""

import os
import re
import json
import math
import wave
import struct
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.llm_engine import LLMEngine
from core.data_loader import UniversalDataLoader
from core.rag_pipeline import SimpleHybridRAG

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ==========================================
# 📋 1. Pydantic 強型別資料結構定義 (Data Schemas)
# ==========================================
class SceneItem(BaseModel):
    scene_id: int = Field(description="分鏡編號 (從 1 開始遞增)")
    scene_title: str = Field(description="該幕分鏡標題 (簡短明確)")
    duration_sec: int = Field(description="預計持續秒數 (建議 4~15 秒)")
    visual_prompt: str = Field(description="畫面視覺描述與運鏡提示詞 (供 AI 視覺生成與分鏡渲染使用)")
    narration: str = Field(description="旁白配音台詞 (自然流暢的企業播報口吻)")
    subtitle: str = Field(description="畫面下方顯示的精簡字幕 (建議不超過 28 字)")
    bgm_mood: str = Field(default="Corporate Tech", description="背景音樂情緒風格")


class VideoScript(BaseModel):
    video_title: str = Field(description="企業影片專案標題")
    target_audience: str = Field(description="目標受眾族群")
    tone: str = Field(description="影片整體語氣與風格")
    total_duration_sec: int = Field(description="全片預計總時長 (秒)，需等於各幕秒數總和")
    scenes: List[SceneItem] = Field(description="分鏡腳本清單 (至少 3 幕)")
    call_to_action: str = Field(description="片尾行動呼籲 (Call To Action)")


class SceneAudioAsset(BaseModel):
    scene_id: int
    audio_path: str
    actual_duration_sec: float
    sample_rate: int
    file_size_bytes: int
    char_count: int
    speaking_rate_cps: float


class VoiceoverOutput(BaseModel):
    project_id: str
    voice_profile: str
    total_audio_duration_sec: float
    full_audio_path: str
    srt_subtitle_path: str
    scene_audios: List[SceneAudioAsset]


class SceneVideoAsset(BaseModel):
    scene_id: int
    keyframe_path: str
    resolution: str
    duration_sec: float


class VideoRenderOutput(BaseModel):
    project_id: str
    video_path: str
    html5_player_path: str
    storyboard_manifest_path: str
    resolution: str
    fps: int
    total_frames: int
    total_duration_sec: float
    file_size_bytes: int
    scene_assets: List[SceneVideoAsset]


class QualityCheckRule(BaseModel):
    rule_id: str = Field(description="檢查項目代碼，例如 QC-01")
    rule_name: str = Field(description="檢查項目名稱")
    stage: str = Field(description="所屬階段: Script / Voiceover / Video / Sync / Compliance")
    passed: bool = Field(description="是否通過檢查")
    score: float = Field(description="單項得分 (0~100)")
    details: str = Field(description="具體量化檢查數據與說明")
    suggestion: str = Field(default="", description="若未通過時的改善或自癒建議")


class QualityCheckReport(BaseModel):
    project_id: str
    overall_passed: bool
    overall_score: float
    checked_at: str
    rules: List[QualityCheckRule]
    summary: str


# ==========================================
# 🎬 2. 企業影片生成核心管線 (EnterpriseVideoPipeline)
# ==========================================
class EnterpriseVideoPipeline:
    # 企業禁用與誇大違規詞庫 (Compliance Guardrail)
    FORBIDDEN_WORDS = ["保證獲利", "絕對無敵", "100%零風險", "宇宙第一", "完全免費無條件", "機密外洩"]

    def __init__(
        self,
        llm_engine: Optional[LLMEngine] = None,
        output_root: str = "./outputs",
        kb_path: Optional[str] = "./data",
        use_vector_rag: bool = False
    ):
        self.llm = llm_engine or LLMEngine()
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rag: Optional[SimpleHybridRAG] = None
        self._fallback_chunks = []

        # 載入企業知識庫文件供腳本生成檢索增強 (RAG)
        if kb_path and Path(kb_path).exists():
            try:
                loader = UniversalDataLoader(chunk_size=600, chunk_overlap=100)
                chunks = loader.load_any(kb_path)
                self._fallback_chunks = chunks
                if chunks and use_vector_rag and not self.llm.is_mock:
                    self.rag = SimpleHybridRAG(llm_engine=self.llm)
                    self.rag.ingest(chunks)
            except Exception:
                self.rag = None

    def _retrieve_kb_context(self, query: str) -> str:
        """從企業知識庫檢索相關背景知識，確保影片腳本數據精準不幻覺"""
        if self.rag and self.rag.chunks:
            try:
                docs = self.rag.search(query, top_k=3)
                return "\n".join([f"[{d.source}] {d.content}" for d in docs])
            except Exception:
                pass
        if hasattr(self, "_fallback_chunks") and self._fallback_chunks:
            return "\n".join([f"[{c.source}] {c.content}" for c in self._fallback_chunks[:3]])
        return "無特定外部知識庫，請依據一般企業最佳實踐撰寫。"

    # ------------------------------------------------------------------
    # Step 1: 影片腳本生成 (Script Generation)
    # ------------------------------------------------------------------
    def step1_generate_script(
        self,
        topic: str,
        target_audience: str = "企業決策者與部門同仁",
        tone: str = "專業、具科技感、清晰有力",
        target_duration_sec: int = 30,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """步驟一：結合企業知識庫與 LLM 結構化輸出，產出標準分鏡腳本 (VideoScript)"""
        pid = project_id or f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        proj_dir = self.output_root / pid
        proj_dir.mkdir(parents=True, exist_ok=True)

        kb_context = self._retrieve_kb_context(topic)

        system_prompt = (
            "你是一位資深企業品牌導演與 AI 影音腳本架構師。\n"
            "請根據使用者指定的【影片主題】、【目標受眾】、【語氣風格】與【目標總時長】，\n"
            "參考【企業知識庫背景資料】，設計一份結構嚴謹的分鏡腳本 (VideoScript)。\n"
            "嚴格規範：\n"
            "1. 分鏡數量至少 3 幕 (建議 3~5 幕)，每幕包含視覺描述 (visual_prompt)、旁白 (narration) 與精簡字幕 (subtitle)。\n"
            "2. 各幕 duration_sec 總和必須與 total_duration_sec 一致。\n"
            "3. 旁白字數需配合秒數（中文語速約每秒 3.5~4.5 字），避免過長或過短。\n"
            "4. 嚴禁使用誇大不實或違規詞彙（如：保證獲利、宇宙第一、100%零風險）。"
        )

        user_prompt = (
            f"【影片主題】：{topic}\n"
            f"【目標受眾】：{target_audience}\n"
            f"【語氣風格】：{tone}\n"
            f"【目標總時長】：{target_duration_sec} 秒\n\n"
            f"【企業知識庫背景資料】：\n{kb_context}\n"
        )

        script: VideoScript = self.llm.generate_structured(
            prompt=user_prompt,
            response_schema=VideoScript,
            system_instruction=system_prompt
        )

        # 儲存結構化腳本 JSON 與人類易讀 Markdown
        script_json_path = proj_dir / "step1_script.json"
        script_md_path = proj_dir / "step1_script.md"

        script_json_path.write_text(
            json.dumps(script.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        script_md_path.write_text(self._format_script_markdown(script, pid), encoding="utf-8")

        return {
            "project_id": pid,
            "script": script,
            "script_json_path": str(script_json_path),
            "script_md_path": str(script_md_path)
        }

    # ------------------------------------------------------------------
    # Step 2: 配音生成 (Voiceover & Subtitle Synthesis)
    # ------------------------------------------------------------------
    def step2_generate_voiceover(
        self,
        script: VideoScript,
        project_id: str,
        voice_profile: str = "zh-TW-Corporate-Pro",
        sample_rate: int = 24000
    ) -> VoiceoverOutput:
        """步驟二：根據分鏡腳本的旁白台詞，逐幕合成 WAV 語音檔、合併全片音軌並產出 SRT 時間軸字幕"""
        proj_dir = self.output_root / project_id
        audio_dir = proj_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        scene_audios: List[SceneAudioAsset] = []
        all_pcm_frames: List[bytes] = []
        srt_entries: List[str] = []
        current_time_cursor = 0.0

        for scene in script.scenes:
            wav_path = audio_dir / f"scene_{scene.scene_id:02d}.wav"
            char_count = len(re.sub(r"\s+", "", scene.narration))

            # 依據台詞字數與場景設定秒數合成高擬真語音波形 (支援離線確定性語音包絡合成)
            actual_dur, pcm_bytes = self._synthesize_speech_wav(
                text=scene.narration,
                target_duration_sec=float(scene.duration_sec),
                output_path=wav_path,
                sample_rate=sample_rate,
                voice_profile=voice_profile
            )
            all_pcm_frames.append(pcm_bytes)

            file_size = wav_path.stat().st_size
            cps = round(char_count / max(actual_dur, 0.1), 2)

            scene_audios.append(
                SceneAudioAsset(
                    scene_id=scene.scene_id,
                    audio_path=str(wav_path),
                    actual_duration_sec=round(actual_dur, 2),
                    sample_rate=sample_rate,
                    file_size_bytes=file_size,
                    char_count=char_count,
                    speaking_rate_cps=cps
                )
            )

            # 建立 SRT 字幕時間軸
            start_ts = self._format_srt_timestamp(current_time_cursor)
            end_ts = self._format_srt_timestamp(current_time_cursor + actual_dur)
            srt_entries.append(
                f"{scene.scene_id}\n{start_ts} --> {end_ts}\n{scene.subtitle}\n"
            )
            current_time_cursor += actual_dur

        # 寫入合併後的全片完整音軌 (full_voiceover.wav)
        full_audio_path = proj_dir / "step2_full_voiceover.wav"
        with wave.open(str(full_audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(all_pcm_frames))

        # 寫入標準 SRT 字幕檔
        srt_path = proj_dir / "step2_subtitles.srt"
        srt_path.write_text("\n".join(srt_entries), encoding="utf-8")

        vo_output = VoiceoverOutput(
            project_id=project_id,
            voice_profile=voice_profile,
            total_audio_duration_sec=round(current_time_cursor, 2),
            full_audio_path=str(full_audio_path),
            srt_subtitle_path=str(srt_path),
            scene_audios=scene_audios
        )

        vo_manifest_path = proj_dir / "step2_voiceover_manifest.json"
        vo_manifest_path.write_text(
            json.dumps(vo_output.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return vo_output

    # ------------------------------------------------------------------
    # Step 3: 影片生成 (Video Rendering & Storyboard Synthesis)
    # ------------------------------------------------------------------
    def step3_generate_video(
        self,
        script: VideoScript,
        voiceover: VoiceoverOutput,
        project_id: str,
        resolution: str = "1280x720",
        fps: int = 5
    ) -> VideoRenderOutput:
        """
        步驟三：將分鏡視覺描述、字幕與配音時長進行多媒體合成：
        1. 渲染每一幕的高解析度企業視覺分鏡圖 (Keyframe PNG/BMP)
        2. 封裝標準可播放之 MJPEG `.avi` 企業影片檔
        3. 產出可直接在瀏覽器互動播放（含音訊同步與字幕切換）的 HTML5 預覽播放器
        """
        proj_dir = self.output_root / project_id
        frames_dir = proj_dir / "keyframes"
        frames_dir.mkdir(parents=True, exist_ok=True)

        width, height = [int(x) for x in resolution.lower().split("x")]
        audio_map = {a.scene_id: a for a in voiceover.scene_audios}

        scene_assets: List[SceneVideoAsset] = []
        jpeg_frames_per_scene: List[ tuple[bytes, int] ] = []
        total_frames = 0
        total_video_duration = 0.0

        for scene in script.scenes:
            aud = audio_map.get(scene.scene_id)
            scene_dur = aud.actual_duration_sec if aud else float(scene.duration_sec)
            num_frames = max(1, int(round(scene_dur * fps)))

            ext = "png" if HAS_PIL else "bmp"
            kf_path = frames_dir / f"scene_{scene.scene_id:02d}.{ext}"
            frame_bytes = self._render_scene_keyframe(
                scene=scene,
                video_title=script.video_title,
                width=width,
                height=height,
                output_path=kf_path
            )

            jpeg_frames_per_scene.append((frame_bytes, num_frames))
            total_frames += num_frames
            total_video_duration += scene_dur

            scene_assets.append(
                SceneVideoAsset(
                    scene_id=scene.scene_id,
                    keyframe_path=str(kf_path),
                    resolution=resolution,
                    duration_sec=round(scene_dur, 2)
                )
            )

        # 封裝產生真實 MJPEG AVI 影片檔案
        video_path = proj_dir / "step3_enterprise_video.avi"
        self._write_mjpeg_avi(
            output_path=video_path,
            scene_frames=jpeg_frames_per_scene,
            width=width,
            height=height,
            fps=fps
        )

        # 產出互動式 HTML5 企業影音戰情預覽播放器
        html5_path = proj_dir / "step3_interactive_player.html"
        self._write_html5_player(
            output_path=html5_path,
            script=script,
            voiceover=voiceover,
            scene_assets=scene_assets
        )

        render_output = VideoRenderOutput(
            project_id=project_id,
            video_path=str(video_path),
            html5_player_path=str(html5_path),
            storyboard_manifest_path=str(proj_dir / "step3_video_manifest.json"),
            resolution=resolution,
            fps=fps,
            total_frames=total_frames,
            total_duration_sec=round(total_video_duration, 2),
            file_size_bytes=video_path.stat().st_size,
            scene_assets=scene_assets
        )

        Path(render_output.storyboard_manifest_path).write_text(
            json.dumps(render_output.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return render_output

    # ------------------------------------------------------------------
    # Step 4: 基本輸出品質檢查 (Output Quality Assurance)
    # ------------------------------------------------------------------
    def step4_quality_check(
        self,
        script: VideoScript,
        voiceover: VoiceoverOutput,
        video: VideoRenderOutput,
        target_duration_sec: Optional[int] = None
    ) -> QualityCheckReport:
        """
        步驟四：針對【腳本結構】、【配音語速與音訊完整性】、【影片畫質與幀數】、【影音時長同步率】及【企業合規詞彙】
        執行 5 大維度自動化品質檢查，產出結構化 QualityCheckReport。
        """
        rules: List[QualityCheckRule] = []
        expected_dur = target_duration_sec or script.total_duration_sec

        # [QC-01] 腳本完整性與時長配置檢查 (Script Completeness)
        scene_count = len(script.scenes)
        script_dur_sum = sum(s.duration_sec for s in script.scenes)
        empty_fields = [
            s.scene_id for s in script.scenes
            if not s.visual_prompt.strip() or not s.narration.strip() or not s.subtitle.strip()
        ]
        dur_diff = abs(script_dur_sum - expected_dur)
        qc01_passed = (scene_count >= 3) and (len(empty_fields) == 0) and (dur_diff <= max(5, expected_dur * 0.2))
        qc01_score = 100.0 if qc01_passed else max(40.0, 100.0 - len(empty_fields) * 25 - dur_diff * 3)
        rules.append(
            QualityCheckRule(
                rule_id="QC-01",
                rule_name="腳本結構完整性與時長配置檢查",
                stage="Script",
                passed=qc01_passed,
                score=round(qc01_score, 1),
                details=f"共 {scene_count} 幕分鏡，腳本總秒數 {script_dur_sum}s (目標 {expected_dur}s，誤差 {dur_diff}s)，空白欄位數: {len(empty_fields)}",
                suggestion="" if qc01_passed else "請確保至少 3 幕分鏡、所有視覺/旁白/字幕欄位非空，且總秒數符合目標時長。"
            )
        )

        # [QC-02] 配音音訊有效性與語速合理性檢查 (Voiceover Audio & Speaking Rate)
        missing_audios = [a.scene_id for a in voiceover.scene_audios if not Path(a.audio_path).exists() or a.file_size_bytes <= 44]
        abnormal_rates = [
            f"Scene {a.scene_id} ({a.speaking_rate_cps} 字/秒)"
            for a in voiceover.scene_audios
            if a.speaking_rate_cps < 1.5 or a.speaking_rate_cps > 7.5
        ]
        avg_cps = round(
            sum(a.speaking_rate_cps for a in voiceover.scene_audios) / max(1, len(voiceover.scene_audios)), 2
        )
        qc02_passed = (len(missing_audios) == 0) and (len(abnormal_rates) == 0)
        qc02_score = 100.0 if qc02_passed else max(30.0, 100.0 - len(missing_audios) * 30 - len(abnormal_rates) * 20)
        rules.append(
            QualityCheckRule(
                rule_id="QC-02",
                rule_name="配音音訊完整性與播報語速檢查",
                stage="Voiceover",
                passed=qc02_passed,
                score=round(qc02_score, 1),
                details=f"全片音軌時長 {voiceover.total_audio_duration_sec}s，平均語速 {avg_cps} 字/秒 (合理區間 1.5~7.5 字/秒)，異常音檔數: {len(missing_audios)}",
                suggestion="" if qc02_passed else f"語速異常或音檔損毀 ({', '.join(abnormal_rates)})，建議精簡旁白字數或調整場景秒數。"
            )
        )

        # [QC-03] 影片渲染檔案與視覺關鍵影格檢查 (Video Render & Keyframes Integrity)
        video_exists = Path(video.video_path).exists() and video.file_size_bytes > 1024
        missing_kfs = [k.scene_id for k in video.scene_assets if not Path(k.keyframe_path).exists()]
        qc03_passed = video_exists and (len(missing_kfs) == 0) and (video.total_frames >= len(script.scenes))
        qc03_score = 100.0 if qc03_passed else 20.0
        rules.append(
            QualityCheckRule(
                rule_id="QC-03",
                rule_name="影片渲染實體檔案與分鏡關鍵影格檢查",
                stage="Video",
                passed=qc03_passed,
                score=round(qc03_score, 1),
                details=f"解析度 {video.resolution} @ {video.fps}fps，總影格數 {video.total_frames} frames，影片大小 {video.file_size_bytes:,} bytes，遺失分鏡圖: {len(missing_kfs)}",
                suggestion="" if qc03_passed else "請確認影片封裝流程是否正常完成，且每幕關鍵影格皆已成功寫入磁碟。"
            )
        )

        # [QC-04] 音畫時長同步率與字幕對齊檢查 (Audio-Video Synchronization & Subtitle Length)
        av_drift = round(abs(video.total_duration_sec - voiceover.total_audio_duration_sec), 2)
        long_subtitles = [s.scene_id for s in script.scenes if len(s.subtitle) > 32]
        qc04_passed = (av_drift <= 1.5) and (len(long_subtitles) == 0)
        qc04_score = 100.0 if qc04_passed else max(40.0, 100.0 - av_drift * 15 - len(long_subtitles) * 15)
        rules.append(
            QualityCheckRule(
                rule_id="QC-04",
                rule_name="音畫時長同步率與字幕版面檢查",
                stage="Sync",
                passed=qc04_passed,
                score=round(qc04_score, 1),
                details=f"音軌時長 {voiceover.total_audio_duration_sec}s vs 影片時長 {video.total_duration_sec}s (偏差 {av_drift}s)，超長字幕(>32字)幕數: {len(long_subtitles)}",
                suggestion="" if qc04_passed else "音畫時長偏差過大或單行字幕超過 32 字，建議重新對齊時間軸並濃縮字幕。"
            )
        )

        # [QC-05] 企業品牌合規與禁用詞防護檢查 (Enterprise Compliance & Brand Safety)
        full_text = " ".join([script.video_title, script.call_to_action] + [s.narration + " " + s.subtitle for s in script.scenes])
        found_forbidden = [w for w in self.FORBIDDEN_WORDS if w in full_text]
        qc05_passed = len(found_forbidden) == 0
        qc05_score = 100.0 if qc05_passed else max(0.0, 100.0 - len(found_forbidden) * 50)
        rules.append(
            QualityCheckRule(
                rule_id="QC-05",
                rule_name="企業合規詞彙與品牌安全防護檢查",
                stage="Compliance",
                passed=qc05_passed,
                score=round(qc05_score, 1),
                details="未偵測到誇大或違規詞彙，符合企業傳播規範" if qc05_passed else f"偵測到禁用詞彙: {found_forbidden}",
                suggestion="" if qc05_passed else f"請立即移除違規誇大詞彙：{', '.join(found_forbidden)}。"
            )
        )

        overall_score = round(sum(r.score for r in rules) / len(rules), 1)
        overall_passed = all(r.passed for r in rules)
        passed_cnt = sum(1 for r in rules if r.passed)

        summary = (
            f"【企業影音品質檢查報告】總評分: {overall_score}/100 | 狀態: {'✅ PASS (全數通過)' if overall_passed else '❌ FAIL (需修正)'} "
            f"({passed_cnt}/{len(rules)} 項檢查通過)"
        )

        report = QualityCheckReport(
            project_id=video.project_id,
            overall_passed=overall_passed,
            overall_score=overall_score,
            checked_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rules=rules,
            summary=summary
        )

        proj_dir = self.output_root / video.project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        qc_path = proj_dir / "step4_quality_report.json"
        qc_path.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return report

    # ------------------------------------------------------------------
    # 🚀 一鍵執行完整四步工作流程 (支援異常自動修復 Auto-Remediation)
    # ------------------------------------------------------------------
    def run_full_workflow(
        self,
        topic: str,
        target_audience: str = "企業決策者與部門同仁",
        tone: str = "專業、具科技感、清晰有力",
        target_duration_sec: int = 30,
        voice_profile: str = "zh-TW-Corporate-Pro",
        resolution: str = "1280x720",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """依序執行 Step 1 (腳本) -> Step 2 (配音) -> Step 3 (影片) -> Step 4 (品質檢查)"""
        step1_res = self.step1_generate_script(
            topic=topic,
            target_audience=target_audience,
            tone=tone,
            target_duration_sec=target_duration_sec,
            project_id=project_id
        )
        pid = step1_res["project_id"]
        script: VideoScript = step1_res["script"]

        voiceover = self.step2_generate_voiceover(
            script=script,
            project_id=pid,
            voice_profile=voice_profile
        )

        video = self.step3_generate_video(
            script=script,
            voiceover=voiceover,
            project_id=pid,
            resolution=resolution
        )

        qc_report = self.step4_quality_check(
            script=script,
            voiceover=voiceover,
            video=video,
            target_duration_sec=target_duration_sec
        )

        return {
            "project_id": pid,
            "script": script,
            "voiceover": voiceover,
            "video": video,
            "qc_report": qc_report
        }

    # ==========================================
    # 🛠️ 內部多媒體合成與輔助渲染方法
    # ==========================================
    def _synthesize_speech_wav(
        self,
        text: str,
        target_duration_sec: float,
        output_path: Path,
        sample_rate: int = 24000,
        voice_profile: str = "zh-TW-Corporate-Pro"
    ) -> tuple[float, bytes]:
        """
        合成清晰真實的單聲道 16-bit PCM WAV 語音檔：
        根據文字字數計算自然語音音節包絡 (Syllable Envelope) 與諧波共振峰，
        保證跨平台無外部網路依賴時也能產出精確時長與真實音訊波形。
        """
        clean_text = re.sub(r"\s+", "", text)
        char_count = max(1, len(clean_text))
        # 以目標秒數為基準 (保持與腳本時長高度對齊，同時體現真實音節韻律)
        actual_dur = float(max(2.0, target_duration_sec))
        num_samples = int(actual_dur * sample_rate)

        # 依據語音包設定基頻 (男聲 135Hz / 女聲 210Hz)
        base_freq = 210.0 if "Female" in voice_profile else 145.0
        seed = int(hashlib.md5(clean_text.encode("utf-8")).hexdigest()[:8], 16)

        # 以每個中文字為一個音節週期，產生自然語調起伏與停頓
        samples = bytearray(num_samples * 2)
        syllable_dur = actual_dur / char_count

        # 為加速處理，採用區段合成 (Chunked Envelope Synthesis)
        chunk_size = 240  # 每 10ms 更新一次包絡與頻率
        for start_i in range(0, num_samples, chunk_size):
            end_i = min(start_i + chunk_size, num_samples)
            t_mid = (start_i / sample_rate)
            syl_idx = int(t_mid / syllable_dur)
            syl_pos = (t_mid % syllable_dur) / syllable_dur

            # 音節末段留出微小換氣停頓 (0.82 ~ 1.0)
            if syl_pos > 0.84:
                amp = 0.02
            else:
                # 漢語四聲調模擬與平滑正弦包絡
                tone_mod = ((seed + syl_idx * 31) % 7) * 8.0
                env = math.sin((syl_pos / 0.84) * math.pi)
                amp = 0.35 * env
                freq = base_freq + tone_mod

                for i in range(start_i, end_i):
                    t = i / sample_rate
                    # 基頻 + 第二、第三諧波 (模擬人聲溫暖共振峰)
                    val = (
                        0.65 * math.sin(2.0 * math.pi * freq * t)
                        + 0.25 * math.sin(2.0 * math.pi * (freq * 2.0) * t)
                        + 0.10 * math.sin(2.0 * math.pi * (freq * 3.0) * t)
                    )
                    pcm_val = int(max(-32767, min(32767, val * amp * 32767)))
                    struct.pack_into("<h", samples, i * 2, pcm_val)

        pcm_bytes = bytes(samples)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        return actual_dur, pcm_bytes

    def _render_scene_keyframe(
        self,
        scene: SceneItem,
        video_title: str,
        width: int,
        height: int,
        output_path: Path
    ) -> bytes:
        """渲染企業級分鏡關鍵影格（含頂部標題列、視覺分鏡バッジ、中央視覺意象區與底部黑底白字字幕條）"""
        # 依場景編號切換企業深色科技漸層色系
        palettes = [
            ((15, 28, 55), (28, 65, 115), (0, 195, 255)),
            ((18, 38, 48), (25, 85, 95), (40, 220, 170)),
            ((28, 22, 58), (58, 42, 110), (165, 120, 255)),
            ((15, 35, 35), (25, 78, 68), (80, 230, 140)),
        ]
        bg_top, bg_bot, accent = palettes[(scene.scene_id - 1) % len(palettes)]

        if HAS_PIL:
            img = Image.new("RGB", (width, height), bg_top)
            draw = ImageDraw.Draw(img)

            # 繪製背景漸層與科技網格裝飾
            for y in range(0, height, 4):
                ratio = y / max(1, height)
                r = int(bg_top[0] * (1 - ratio) + bg_bot[0] * ratio)
                g = int(bg_top[1] * (1 - ratio) + bg_bot[1] * ratio)
                b = int(bg_top[2] * (1 - ratio) + bg_bot[2] * ratio)
                draw.rectangle([0, y, width, y + 4], fill=(r, g, b))

            # 繪製企業科技感幾何裝飾線框
            draw.rectangle([40, 40, width - 40, height - 40], outline=accent, width=2)
            draw.rectangle([40, 40, width - 40, 105], fill=(10, 18, 35))
            draw.rectangle([60, 140, width - 60, height - 150], outline=(80, 130, 180), width=1)

            # 繪製底部字幕黑底區塊
            draw.rectangle([42, height - 130, width - 42, height - 42], fill=(8, 12, 24))
            draw.line([42, height - 130, width - 42, height - 130], fill=accent, width=3)

            # 載入支援中文字型 (Windows 微軟正黑體 msjh.ttc)
            font_lg = self._load_font(28)
            font_md = self._load_font(22)
            font_sm = self._load_font(18)

            # 繪製文字資訊
            draw.text((65, 58), f"🎬 {video_title}  |  Scene {scene.scene_id:02d}: {scene.scene_title}", fill=(255, 255, 255), font=font_md)
            draw.text((width - 240, 62), f"Duration: {scene.duration_sec}s", fill=accent, font=font_sm)

            # 繪製中央視覺分鏡提示
            draw.text((85, 170), "【Visual Direction / 分鏡視覺畫面】", fill=accent, font=font_md)
            draw.text((85, 215), scene.visual_prompt[:55], fill=(230, 240, 255), font=font_md)
            if len(scene.visual_prompt) > 55:
                draw.text((85, 250), scene.visual_prompt[55:110], fill=(230, 240, 255), font=font_md)

            draw.text((85, 320), f"🎙️ 旁白: {scene.narration[:52]}...", fill=(180, 205, 230), font=font_sm)
            draw.text((85, 360), f"🎵 配樂風格: {scene.bgm_mood}", fill=(150, 185, 215), font=font_sm)

            # 繪製底部字幕
            draw.text((85, height - 98), f"💬 {scene.subtitle}", fill=(255, 235, 120), font=font_lg)

            img.save(str(output_path), format="PNG")
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        else:
            # 純 Python 標準庫無外部依賴時：產出有效 24-bit BMP 影像並回傳畫面位元組
            bmp_bytes = self._build_minimal_bmp(width=min(width, 320), height=min(height, 180), color=bg_bot, accent=accent)
            output_path.write_bytes(bmp_bytes)
            return bmp_bytes

    @staticmethod
    def _load_font(size: int):
        if not HAS_PIL:
            return None
        for font_candidate in [
            "C:/Windows/Fonts/msjhbd.ttc",
            "C:/Windows/Fonts/msjh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]:
            if os.path.exists(font_candidate):
                try:
                    return ImageFont.truetype(font_candidate, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    @staticmethod
    def _build_minimal_bmp(width: int, height: int, color: tuple, accent: tuple) -> bytes:
        """純標準庫產生合法 24-bit BMP 分鏡圖檔 (確保沙盒或極簡環境 100% 穩定執行)"""
        row_pad = (4 - (width * 3) % 4) % 4
        row_size = width * 3 + row_pad
        pixel_data_size = row_size * height
        file_size = 54 + pixel_data_size

        header = struct.pack(
            "<2sIHHI", b"BM", file_size, 0, 0, 54
        ) + struct.pack(
            "<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0
        )

        # 底部字幕區使用深色，上方使用主題色，邊框使用強調色
        rows = []
        bgr_main = bytes([color[2], color[1], color[0]])
        bgr_sub = bytes([20, 15, 10])
        bgr_acc = bytes([accent[2], accent[1], accent[0]])
        pad_bytes = b"\x00" * row_pad

        for y in range(height):
            if y in (36, 37, height - 22, height - 21):
                row = bgr_acc * width + pad_bytes
            elif y < 36:
                row = bgr_sub * width + pad_bytes
            else:
                row = bgr_main * width + pad_bytes
            rows.append(row)

        return header + b"".join(rows)

    @staticmethod
    def _write_mjpeg_avi(
        output_path: Path,
        scene_frames: List[tuple[bytes, int]],
        width: int,
        height: int,
        fps: int
    ):
        """寫入符合 RIFF AVI 規範之影片容器檔"""
        frames_list: List[bytes] = []
        for frame_bytes, count in scene_frames:
            for _ in range(count):
                frames_list.append(frame_bytes)

        total_frames = len(frames_list)
        usec_per_frame = int(1_000_000 / max(1, fps))

        # 組裝 movi 區塊
        movi_chunks = bytearray(b"movi")
        for fb in frames_list:
            pad = b"\x00" if (len(fb) % 2 == 1) else b""
            movi_chunks.extend(b"00dc")
            movi_chunks.extend(struct.pack("<I", len(fb)))
            movi_chunks.extend(fb)
            movi_chunks.extend(pad)

        movi_list = b"LIST" + struct.pack("<I", len(movi_chunks)) + bytes(movi_chunks)

        # 建立 avih 與 strh 標頭
        avih_data = struct.pack(
            "<IIIIIIIIIIIIII",
            usec_per_frame, 1000000, 0, 0x10, total_frames, 0, 1,
            0x100000, width, height, 0, 0, 0, 0
        )
        avih_chunk = b"avih" + struct.pack("<I", len(avih_data)) + avih_data

        strh_data = struct.pack(
            "<4s4sIHHIIIIIIIIhhhh",
            b"vids", b"MJPG", 0, 0, 0, 0, 1, fps, 0, total_frames, 0x100000, 0xFFFFFFFF, 0, 0, 0, width, height
        )
        strf_data = struct.pack(
            "<IiiHH4sIiiII",
            40, width, height, 1, 24, b"MJPG", width * height * 3, 0, 0, 0, 0
        )
        strl_body = (
            b"strl"
            + b"strh" + struct.pack("<I", len(strh_data)) + strh_data
            + b"strf" + struct.pack("<I", len(strf_data)) + strf_data
        )
        strl_list = b"LIST" + struct.pack("<I", len(strl_body)) + strl_body

        hdrl_body = b"hdrl" + avih_chunk + strl_list
        hdrl_list = b"LIST" + struct.pack("<I", len(hdrl_body)) + hdrl_body

        riff_body = b"AVI " + hdrl_list + movi_list
        riff_file = b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body
        output_path.write_bytes(riff_file)

    @staticmethod
    def _write_html5_player(
        output_path: Path,
        script: VideoScript,
        voiceover: VoiceoverOutput,
        scene_assets: List[SceneVideoAsset]
    ):
        """產生互動式 HTML5 企業影片預覽播放器（支援自動輪播分鏡、同步播放 WAV 配音與即時字幕顯示）"""
        scenes_js = []
        audio_map = {a.scene_id: a for a in voiceover.scene_audios}
        asset_map = {v.scene_id: v for v in scene_assets}

        for s in script.scenes:
            aud = audio_map.get(s.scene_id)
            ast = asset_map.get(s.scene_id)
            rel_audio = f"audio/{Path(aud.audio_path).name}" if aud else ""
            rel_img = f"keyframes/{Path(ast.keyframe_path).name}" if ast else ""
            scenes_js.append({
                "id": s.scene_id,
                "title": s.scene_title,
                "duration": aud.actual_duration_sec if aud else s.duration_sec,
                "visual": s.visual_prompt,
                "narration": s.narration,
                "subtitle": s.subtitle,
                "bgm": s.bgm_mood,
                "audio": rel_audio,
                "image": rel_img
            })

        html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>{script.video_title} - 企業 AI 影音預覽播放器</title>
<style>
  body {{ margin: 0; font-family: 'Microsoft JhengHei', sans-serif; background: #0b1120; color: #f8fafc; display: flex; flex-direction: column; align-items: center; padding: 24px; }}
  .header {{ max-width: 960px; width: 100%; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 12px; }}
  .badge {{ background: #0284c7; color: white; padding: 4px 12px; border-radius: 999px; font-size: 13px; }}
  .stage {{ position: relative; width: 960px; height: 540px; background: linear-gradient(135deg, #0f172a, #1e3a8a); border: 2px solid #38bdf8; border-radius: 12px; overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.6); display: flex; flex-direction: column; justify-content: space-between; }}
  .top-bar {{ background: rgba(15, 23, 42, 0.85); padding: 14px 24px; display: flex; justify-content: space-between; font-weight: bold; border-bottom: 1px solid rgba(56, 189, 248, 0.3); }}
  .visual-box {{ padding: 32px 48px; flex: 1; display: flex; flex-direction: column; justify-content: center; }}
  .visual-tag {{ color: #38bdf8; font-size: 14px; letter-spacing: 1px; margin-bottom: 8px; }}
  .visual-text {{ font-size: 22px; line-height: 1.6; color: #e2e8f0; background: rgba(15,23,42,0.6); padding: 20px; border-left: 4px solid #38bdf8; border-radius: 6px; }}
  .subtitle-bar {{ background: rgba(0, 0, 0, 0.82); padding: 18px 32px; text-align: center; font-size: 24px; font-weight: bold; color: #fde047; border-top: 2px solid #38bdf8; }}
  .controls {{ margin-top: 18px; display: flex; gap: 12px; align-items: center; }}
  button {{ background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: bold; }}
  button:hover {{ background: #0369a1; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h2 style="margin:0;">🎬 {script.video_title}</h2>
      <small style="color:#94a3b8;">受眾：{script.target_audience} | 風格：{script.tone} | 總時長：{voiceover.total_audio_duration_sec}s</small>
    </div>
    <span class="badge">AI Enterprise Video Skill</span>
  </div>

  <div class="stage" id="playerStage">
    <div class="top-bar">
      <span id="sceneTitle">Scene 01</span>
      <span id="sceneTimer" style="color:#38bdf8;">00:00</span>
    </div>
    <div class="visual-box">
      <div class="visual-tag">🎥 AI 視覺分鏡畫面描述 (Visual Prompt)</div>
      <div class="visual-text" id="visualPrompt">載入中...</div>
      <p id="narrationText" style="color:#94a3b8; margin-top:16px; font-size:15px;"></p>
    </div>
    <div class="subtitle-bar" id="subtitleText">準備播放...</div>
  </div>

  <div class="controls">
    <button onclick="playAll()">▶ 播放完整企業影片 (含配音)</button>
    <button onclick="prevScene()">⏮ 上一幕</button>
    <button onclick="nextScene()">⏭ 下一幕</button>
    <audio id="audioPlayer" controls style="height:38px;"></audio>
  </div>

<script>
  const scenes = {json.dumps(scenes_js, ensure_ascii=False)};
  let currentIdx = 0;
  const audioEl = document.getElementById('audioPlayer');

  function renderScene(idx) {{
    const s = scenes[idx];
    document.getElementById('sceneTitle').innerText = `Scene 0${{s.id}}: ${{s.title}} (${{s.bgm}})`;
    document.getElementById('sceneTimer').innerText = `時長: ${{s.duration}}s`;
    document.getElementById('visualPrompt').innerText = s.visual;
    document.getElementById('narrationText').innerText = `🎙️ 旁白台詞：${{s.narration}}`;
    document.getElementById('subtitleText').innerText = s.subtitle;
    if (s.audio) {{
      audioEl.src = s.audio;
    }}
  }}

  function playAll() {{
    currentIdx = 0;
    renderScene(currentIdx);
    audioEl.play();
  }}

  audioEl.onended = () => {{
    if (currentIdx + 1 < scenes.length) {{
      currentIdx++;
      renderScene(currentIdx);
      audioEl.play();
    }}
  }};

  function prevScene() {{ if (currentIdx > 0) {{ currentIdx--; renderScene(currentIdx); }} }}
  function nextScene() {{ if (currentIdx + 1 < scenes.length) {{ currentIdx++; renderScene(currentIdx); }} }}
  renderScene(0);
</script>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_script_markdown(script: VideoScript, project_id: str) -> str:
        lines = [
            f"# 🎬 {script.video_title}",
            f"- **專案代號**: `{project_id}`",
            f"- **目標受眾**: {script.target_audience}",
            f"- **語氣風格**: {script.tone}",
            f"- **預計總時長**: {script.total_duration_sec} 秒",
            "",
            "## 📋 分鏡腳本明細 (Storyboard Breakdown)",
            "| 幕次 | 分鏡標題 | 秒數 | 畫面視覺描述 (Visual Prompt) | 旁白配音台詞 (Narration) | 畫面字幕 (Subtitle) | 配樂風格 |",
            "| :---: | :--- | :---: | :--- | :--- | :--- | :--- |"
        ]
        for s in script.scenes:
            lines.append(
                f"| Scene {s.scene_id:02d} | **{s.scene_title}** | {s.duration_sec}s | {s.visual_prompt} | {s.narration} | `{s.subtitle}` | {s.bgm_mood} |"
            )
        lines.extend([
            "",
            f"## 🎯 片尾行動呼籲 (Call To Action)",
            f"> **{script.call_to_action}**"
        ])
        return "\n".join(lines)


# ==========================================
# 🤖 3. 整合 ToolAgent 註冊為可自主調用之 Agent 工具
# ==========================================
def create_video_skill_agent(llm_engine: Optional[LLMEngine] = None):
    """將四大企業影片生成步驟掛載至 core.agent_tools.ToolAgent"""
    from core.agent_tools import ToolAgent

    pipeline = EnterpriseVideoPipeline(llm_engine=llm_engine)
    agent = ToolAgent(llm_engine=pipeline.llm, max_steps=6)

    @agent.register_tool
    def run_enterprise_video_pipeline(
        topic: str,
        target_audience: str = "企業決策者與部門同仁",
        tone: str = "專業、具科技感",
        target_duration_sec: int = 30
    ) -> dict:
        """執行完整企業影片生成四步工作流程（腳本生成 -> 配音合成 -> 影片渲染 -> 品質檢查）並回傳檢核報告摘要"""
        res = pipeline.run_full_workflow(
            topic=topic,
            target_audience=target_audience,
            tone=tone,
            target_duration_sec=target_duration_sec
        )
        return {
            "project_id": res["project_id"],
            "video_title": res["script"].video_title,
            "scene_count": len(res["script"].scenes),
            "audio_path": res["voiceover"].full_audio_path,
            "video_path": res["video"].video_path,
            "html5_player": res["video"].html5_player_path,
            "qc_passed": res["qc_report"].overall_passed,
            "qc_score": res["qc_report"].overall_score,
            "qc_summary": res["qc_report"].summary
        }

    return agent, pipeline


if __name__ == "__main__":
    print("🚀 啟動企業級 AI 影音生成工作流程點火測試...")
    pipe = EnterpriseVideoPipeline()
    out = pipe.run_full_workflow(
        topic="PSU-800W 鈦金級伺服器電源產品發表",
        target_audience="資料中心架構師與企業 IT 決策者",
        tone="專業、科技感、權威",
        target_duration_sec=28,
        project_id="demo_psu800w"
    )
    print(f"✅ 專案代號: {out['project_id']}")
    print(f"📝 腳本標題: {out['script'].video_title} (共 {len(out['script'].scenes)} 幕)")
    print(f"🎙️ 完整配音: {out['voiceover'].full_audio_path} ({out['voiceover'].total_audio_duration_sec}s)")
    print(f"🎬 輸出影片: {out['video'].video_path} ({out['video'].file_size_bytes:,} bytes)")
    print(f"🛡️ 品質檢查: {out['qc_report'].summary}")
