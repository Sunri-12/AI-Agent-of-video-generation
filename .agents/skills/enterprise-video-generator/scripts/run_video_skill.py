#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Video Generator Skill - CLI Runner
支援完整四步管線執行 (--mode full) 或單步拆解執行 (--mode step1 / step2 / step3 / step4)
"""

import sys
import json
import argparse
from pathlib import Path

# 將專案根目錄加入 sys.path
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.video_workflow import (
    EnterpriseVideoPipeline,
    VideoScript,
    VoiceoverOutput,
    VideoRenderOutput,
)


def main():
    parser = argparse.ArgumentParser(description="Enterprise AI Video Generation Skill CLI")
    parser.add_argument(
        "--mode",
        choices=["full", "step1", "step2", "step3", "step4", "agent"],
        default="full",
        help="執行模式: full(完整四步驟), step1(腳本), step2(配音), step3(影片), step4(品質檢查), agent(ReAct Agent 自主調用)"
    )
    parser.add_argument("--topic", type=str, default="PSU-800W 鈦金級伺服器電源產品發表", help="企業影片主題")
    parser.add_argument("--audience", type=str, default="企業決策者與技術團隊", help="目標受眾")
    parser.add_argument("--tone", type=str, default="專業、具科技感、權威", help="語氣風格")
    parser.add_argument("--duration", type=int, default=28, help="預計影片總秒數")
    parser.add_argument("--voice", type=str, default="zh-TW-Corporate-Pro", help="配音語音模型設定")
    parser.add_argument("--resolution", type=str, default="1280x720", help="輸出影片解析度")
    parser.add_argument("--project-id", type=str, default="enterprise_demo", help="輸出專案資料夾代號")
    parser.add_argument("--output-root", type=str, default=str(ROOT_DIR / "outputs"), help="輸出根目錄")

    args = parser.parse_args()
    pipeline = EnterpriseVideoPipeline(output_root=args.output_root, kb_path=str(ROOT_DIR / "data"))
    proj_dir = Path(args.output_root) / args.project_id

    if args.mode == "full":
        print(f"🎬 [Full Workflow] 開始執行企業影片生成管線 | 專案: {args.project_id}")
        res = pipeline.run_full_workflow(
            topic=args.topic,
            target_audience=args.audience,
            tone=args.tone,
            target_duration_sec=args.duration,
            voice_profile=args.voice,
            resolution=args.resolution,
            project_id=args.project_id
        )
        print(f"✅ [Step 1 腳本生成] {res['script'].video_title} ({len(res['script'].scenes)} 幕)")
        print(f"✅ [Step 2 配音合成] {res['voiceover'].full_audio_path} ({res['voiceover'].total_audio_duration_sec}s)")
        print(f"✅ [Step 3 影片生成] {res['video'].video_path} ({res['video'].file_size_bytes:,} bytes)")
        print(f"✅ [Step 4 品質檢查] {res['qc_report'].summary}")
        return

    if args.mode == "step1":
        res = pipeline.step1_generate_script(
            topic=args.topic,
            target_audience=args.audience,
            tone=args.tone,
            target_duration_sec=args.duration,
            project_id=args.project_id
        )
        print(f"✅ [Step 1 完成] 腳本已存至: {res['script_json_path']}")

    elif args.mode == "step2":
        script_path = proj_dir / "step1_script.json"
        if not script_path.exists():
            raise FileNotFoundError(f"找不到 Step 1 腳本檔案: {script_path}，請先執行 --mode step1")
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        vo = pipeline.step2_generate_voiceover(script=script, project_id=args.project_id, voice_profile=args.voice)
        print(f"✅ [Step 2 完成] 完整配音檔: {vo.full_audio_path} | 字幕檔: {vo.srt_subtitle_path}")

    elif args.mode == "step3":
        script_path = proj_dir / "step1_script.json"
        vo_path = proj_dir / "step2_voiceover_manifest.json"
        script = VideoScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        vo = VoiceoverOutput.model_validate_json(vo_path.read_text(encoding="utf-8"))
        vid = pipeline.step3_generate_video(script=script, voiceover=vo, project_id=args.project_id, resolution=args.resolution)
        print(f"✅ [Step 3 完成] 企業影片檔: {vid.video_path} | HTML5 預覽器: {vid.html5_player_path}")

    elif args.mode == "step4":
        script = VideoScript.model_validate_json((proj_dir / "step1_script.json").read_text(encoding="utf-8"))
        vo = VoiceoverOutput.model_validate_json((proj_dir / "step2_voiceover_manifest.json").read_text(encoding="utf-8"))
        vid = VideoRenderOutput.model_validate_json((proj_dir / "step3_video_manifest.json").read_text(encoding="utf-8"))
        qc = pipeline.step4_quality_check(script=script, voiceover=vo, video=vid, target_duration_sec=args.duration)
        print(f"✅ [Step 4 完成] {qc.summary}")
        for r in qc.rules:
            status = "PASS" if r.passed else "FAIL"
            print(f"  - [{r.rule_id}] {r.rule_name}: {status} ({r.score}分) -> {r.details}")

    elif args.mode == "agent":
        from core.video_workflow import create_video_skill_agent
        agent, _ = create_video_skill_agent()
        query = f"請幫我製作一支主題為「{args.topic}」的企業影片，目標受眾是「{args.audience}」，時長約 {args.duration} 秒，並回報品質檢查結果。"
        agent.run(query, verbose=True)


if __name__ == "__main__":
    main()
