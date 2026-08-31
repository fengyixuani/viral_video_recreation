"""core —— 参考字幕特效复刻的实现模块（从 Viral_Video_Agent 的 whq_clone/captions_clone 抽出）。

对外只用 `caption_clone.py` 的 `clone_captions()`；这里的模块按流水线分工：

    frames        抽帧（VLM 用小图 + 像素校准用原生大图）
    ref_analyzer  VLM 逐帧读参考视频的字幕（文字/bbox/位置/颜色/字号/特效/角色）
    color_calib   在原生帧上像素级取色，校准 VLM 目测色（PIL + numpy，缺了就跳过）
    style_profile 把逐帧清单蒸成风格档（密度/配色/位置/特效偏好）
    profile_cache 参考风格分析 + 按「参考视频指纹 + 抽帧率 + 模型」缓存
    asr_tokens    对目标成片跑词级 ASR（逐字时间戳）
    charstream    逐字流 → 逐句念白（有 plan 用 plan 文本，没有就用 ASR 文本）
    target_match  LLM 按参考风格把念白拆块、配色配位配特效、标关键词；同音错字纠错
    ass_burn      生成 ASS 并用 ffmpeg/libass 烧进视频
    inventory_md  参考字幕清单 markdown（人看的中间产物）
    gateway       LLM/VLM 网关调用 + JSON 兜底 + 阶段缓存（替代 Agent 的 pipeline_utils/as_core）
    _env          FFMPEG/FFPROBE 解析（替代 Agent 的 _common）
"""
