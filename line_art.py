"""真人人脸规避能力：seedance 拒收含真人人脸的参考图/首帧（错误码 40000002）。

做法：用纯文生图画一张「写实身体 + 脸部黑白线稿」的人物图当参考图，
既过风控，又保住「看起来是同一个人」的信息（发型/服装/体型/配色）——
这些外观信息由 VLM 从原图读成文字描述，再拼进 t2i prompt。

典型用途：
- 角色形象参考图：同一角色跨镜头一致
- 桥接帧：不能直接拿上一段视频的写实真人尾帧去续下一段，改为按 end_state 生成一张线描帧
"""
import os

import aigc
import storage

# prompt 拼装：_REDRAW_BASE（构图/画质）+ 人物外观描述 + FACE_LINE_ART（线稿与写实的材质边界）。
# FACE_LINE_ART 走「材质拼接」写法：面部是独立矢量介质，发际线与颈部是写实分界带，
# 靠介质互斥把线稿封在面部范围内——只说"脸是线稿、身体写实"会让线稿扩散到衣服和全身。
# 线稿范围只能是五官所在的面部，不能含头发：把头发也线稿化会画成白色线条头发，
# 发色这条一致性信息就丢了，成片里人物会顶着一头白发。
FACE_LINE_ART = (
    "五官所在的面部区域呈现为纯白底黑色细线矢量轮廓图，只画眉眼鼻嘴与脸部轮廓线条，"
    "绝对无肤色无阴影无纹理；头发按上述发型发色以真实毛发质感写实渲染，有光泽与发丝细节，"
    "不做任何线稿化处理；发际线与颈部作为明确材质分割带，颈部锁骨向下为真实人类肉体与衣物，"
    "皮肤具备毛孔光泽与血色质感，服装为哑光棉质实拍纹理，鞋子为皮革反光细节。"
    "确保黑白线稿严格封闭在面部五官范围内，形成面部平面线稿与写实头发躯干的精准拼贴效果，"
    "自然光均匀照明。"
)

# 只写「4K写实摄影」压不住风格，约 2/3 的图会出插画/动漫质感。前缀换成具体的相机光学参数
# 加「实体照片扫描件」的介质声明后，实测 12/12 判为实拍且线稿脸不受影响；
# 抽象的「不是插画」、场景叙事式（电商实拍图）、以及两者叠加都更差。
# 出处 output/real_photo_prefix/summary.json（p7_camera_scan 12/12，基线 p0 4/12）。
_REDRAW_BASE = (
    "数码单反相机实拍照片，85mm 定焦镜头，f/2.8 光圈，ISO 200，1/160s 快门，"
    "影棚柔光箱三点布光，浅景深，正面全身站立人物，纯色背景。"
    "画面为实体照片扫描件，所有毛发、皮肤、布料均为光学成像的真实材质，"
    "无任何手绘或渲染成分。"
)

_DESCRIBE_PROMPT = (
    "用一段中文描述这张图里人物的外观，只写：性别与大致年龄、发型发色、上衣/下装/鞋子的款式与颜色、"
    "体型、随身道具。不要描述五官、表情、背景和光线，不要评价，不超过80字，直接输出描述。"
)


def describe_person(src: str) -> str:
    """把参考图里的人物外观转成文字（发型/服装/体型/配色），供 t2i 复现同一个人。"""
    url = storage.upload(src) if os.path.isfile(src) else src
    return aigc.vision(_DESCRIBE_PROMPT, media=[{"type": "image", "url": url}]).strip()


def make_eye_occluded_person(src: str, out_path: str,
                             style: str = "sunglasses") -> str:
    """只遮挡双眼，保留原图其它像素，生成 Seedance 对照参考图。

    遮挡位置按 InsightFace 的标准 68 点关键点和眼间距计算，换分辨率
    或换人物时不需要修改坐标。``sunglasses`` 是默认的自然遮挡样式，
    ``eye_band`` 用于对照测试更直接的眼部遮挡。
    """
    import cv2
    import numpy as np

    image = cv2.imread(os.path.abspath(src))
    if image is None:
        raise RuntimeError("无法读取本地参考图：%s" % src)

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=os.getenv(
        "INSIGHTFACE_ROOT", "/root/.insightface"))
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(image)
    if not faces:
        raise RuntimeError("本地参考图未检测到人脸")
    face = max(
        faces,
        key=lambda item: max(1.0, float(item.bbox[2] - item.bbox[0]))
        * max(1.0, float(item.bbox[3] - item.bbox[1])),
    )
    points = getattr(face, "landmark_3d_68", None)
    if points is None or len(points) < 68:
        raise RuntimeError("人脸检测未返回 68 点关键点")
    points = np.asarray(points, dtype=np.float32)[:, :2]
    eyes = [points[36:42].mean(axis=0), points[42:48].mean(axis=0)]
    eye_distance = float(np.linalg.norm(eyes[1] - eyes[0]))
    if eye_distance < 10:
        raise RuntimeError("人脸关键点尺度异常")

    style = (style or "sunglasses").strip().lower()
    if style not in ("sunglasses", "eye_band"):
        raise ValueError("未知眼部遮挡样式：%s" % style)

    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    angle = float(np.degrees(np.arctan2(
        eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0])))
    thickness = max(2, int(round(0.018 * eye_distance)))

    if style == "sunglasses":
        axes = (
            max(8, int(round(0.34 * eye_distance))),
            max(6, int(round(0.20 * eye_distance))),
        )
        for center in eyes:
            c = tuple(int(round(v)) for v in center)
            cv2.ellipse(mask, c, axes, angle, 0, 360, 255, -1)
            cv2.ellipse(overlay, c, axes, angle, 0, 360,
                        (28, 32, 36), -1)
            cv2.ellipse(overlay, c, axes, angle, 0, 360,
                        (8, 10, 12), thickness)
        cv2.line(
            overlay,
            tuple(int(round(v)) for v in eyes[0]),
            tuple(int(round(v)) for v in eyes[1]),
            (8, 10, 12), thickness)
    else:
        left = points[36:42].min(axis=0)
        right = points[42:48].max(axis=0)
        x0 = int(round(left[0] - 0.14 * eye_distance))
        x1 = int(round(right[0] + 0.14 * eye_distance))
        cy = int(round((eyes[0][1] + eyes[1][1]) * 0.5))
        half_h = max(8, int(round(0.25 * eye_distance)))
        cv2.rectangle(mask, (x0, cy - half_h), (x1, cy + half_h), 255, -1)
        cv2.rectangle(overlay, (x0, cy - half_h), (x1, cy + half_h),
                      (58, 62, 66), -1)

    alpha = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0, (0, 0),
        sigmaX=max(1.0, 0.012 * eye_distance))[..., None]
    output = (
        image.astype(np.float32) * (1.0 - alpha)
        + overlay.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not cv2.imwrite(out_path, output):
        raise RuntimeError("无法写出眼部遮挡参考图：%s" % out_path)
    return out_path


# 40000002 是通用的「参数非法」码，风控拒绝和参数错误共用它，所以只能按错误文案判。
# 按码判会把「参考视频短于 1.8s」这类参数错误误判成风控，白跑三次降级还把真因埋掉。
REAL_PERSON_ERRORS = ("real person", "OutputImageSensitiveContentDetected",
                      "InputImageSensitiveContentDetected", "sensitive", "风控", "审核")


def is_real_person_reject(exc) -> bool:
    """异常是否为真人风控拒绝。"""
    text = str(exc).lower()
    return any(k.lower() in text for k in REAL_PERSON_ERRORS)


def to_line_art(src: str, extra_prompt: str = "", size: str = None) -> str:
    """把含真人人脸的图换成一张「写实身体 + 脸部线稿」图，返回可安全喂 seedance 的 URL。
    src 为本地路径或公网 URL，只用来读外观描述，图本身不进 seedream。
    """
    return gen_line_art_frame(describe_person(src) + (extra_prompt or ""), size=size)


# 服装、配饰这类商品，商品图本身就是「模特穿着实拍」：整张图重画（to_line_art）会把商品丢掉，
# 原样保留又过不了真人风控，只能退到纯文生视频——商品信息一样全丢。所以对这类图只动脸：
# 面部换成线稿贴纸，服装/姿态/背景全部保持原像素，再配 REAL_ACTOR_HINT 让成片长回真人脸。
_HAS_PERSON_PROMPT = ('这张图里有没有出现真人的脸或身体？只回答一个字："有" 或 "无"。')

_FACE_SAFE_PROMPT = (
    "只改人物的面部：把五官所在的面部区域替换为纯白底黑色细线的矢量线稿贴纸，"
    "只留眉眼鼻嘴与脸部轮廓线条，无肤色无阴影。"
    "画面其余部分必须与原图完全一致：发型发色写实不变，服装的款式、版型、颜色、图案、面料质感、"
    "褶皱走向一模一样，身材姿态、手部、鞋子、背景、构图与光线全部保持原样，不要改动任何商品细节。")

_face_safe_cache: dict = {}
_face_safe_lock = None


def has_real_person(src: str) -> bool:
    """图里有没有真人（脸或身体）。判不出时按「有」处理：宁可多改一张脸，也别撞风控白跑。"""
    url = storage.upload(src) if os.path.isfile(src) else src
    try:
        got = aigc.understand(_HAS_PERSON_PROMPT, media=[{"type": "image", "url": url}],
                              max_tokens=16)
    except Exception:  # noqa: BLE001
        return True
    return "无" not in str(got)


def to_face_safe(src: str, size: str = None) -> str:
    """商品图里有真人时，只把脸换成线稿贴纸，商品（服装等）与画面其余部分保持原样。

    没有真人就原样返回 src——纯商品图（手机、牙刷）不需要动，改一遍反而可能改坏商品。
    同一张图在多段生成里会被反复用到，结果按 (URL, size) 缓存，只改一次——
    size 决定出图分辨率，漏进 key 的话同一张图先按 A 尺寸生成、后按 B 尺寸请求时会拿到 A 的结果。
    """
    global _face_safe_lock
    if _face_safe_lock is None:
        import threading
        _face_safe_lock = threading.Lock()
    key = (src, size or "")
    with _face_safe_lock:
        if key in _face_safe_cache:
            return _face_safe_cache[key]
    out = src
    try:
        if has_real_person(src):
            out = aigc.gen_image(_FACE_SAFE_PROMPT, size=size, ref_images=[src])
    except Exception:  # noqa: BLE001
        out = src            # 改脸失败就退回原图，让上层继续按原顺序降级
    with _face_safe_lock:
        _face_safe_cache[key] = out
    return out


def gen_line_art_frame(desc: str, size: str = None, ref_images: list = None) -> str:
    """按文字描述生成一张「脸部线描」画面，用于角色参考图 / 段间桥接帧。
    desc 为人物外观与这一帧的画面描述。
    ref_images 传商品原图时走参考图生图，保证画面里的商品与真实商品一致；
    不要传真人照片，生成图里出现写实人脸会被 seedance 风控拒。
    """
    return aigc.gen_image(_REDRAW_BASE + desc.strip().rstrip("。") + "。" + FACE_LINE_ART,
                          size=size, ref_images=ref_images)


# 线稿参考图会被 seedance 照抄进成片（人物顶着一张白色线稿脸）。REAL_ACTOR_HINT 把线稿脸
# 显式定义成「需要被替换的占位符」，前置在 prompt 最前面，成片才会渲染成正常真人脸。
# 出处 output/seedance_person/history.jsonl，5/5 实测通过。
REAL_ACTOR_HINT = (
    "生成真实人类面部替换参考图中的线稿贴纸占位符，呈现完整五官、自然肤色与写实质感。"
    "人物保持参考图的发型发色、服装与体型不变。4K画质，电影级色彩分级，"
    "确保人脸真实细腻无畸变，线稿元素完全消失仅作为结构引导。"
)


def gen_video_safe(prompt: str, ref_images=None, line_art_extra: str = "",
                   keep_refs=None, **kw) -> dict:
    """带真人风控自动降级的视频生成，返回 {"video_url", "mode", "ref_images"}。
    回退顺序：原参考图(ref2v) → 参考图逐张换成线描图后重试(line_art) → 丢参考图纯文生视频(t2v)。
    走 line_art 时会把 REAL_ACTOR_HINT 前置到 prompt 最前面，否则线稿脸会被照抄进成片。
    keep_refs（商品图）不整张转线稿：to_line_art 走 describe_person 只保留人物外观信息，
    商品图过一遍会被改造成一张人物图，商品信息全丢；但商品图本身是模特实拍（服装类）时
    原样保留过不了风控，所以对它们走 to_face_safe——只换脸，商品与画面其余部分不动。
    只在被真人风控拒时才降级，其它错误直接抛。kw 透传给 aigc.gen_video。
    """
    refs = list(ref_images or [])
    keep = set(keep_refs or [])
    attempts = [("ref2v", refs), ("line_art", None), ("t2v", [])] if refs else [("t2v", [])]

    last = None
    for mode, urls in attempts:
        try:
            p = prompt
            if mode == "line_art":
                urls = [to_face_safe(u) if u in keep else to_line_art(u, line_art_extra)
                        for u in refs]
                if urls == refs:   # 全是无真人的商品图，没什么可改，重试也不会有变化
                    continue
                p = REAL_ACTOR_HINT + prompt
            return {"video_url": aigc.gen_video(p, ref_images=urls or None, **kw),
                    "mode": mode, "ref_images": urls or []}
        except RuntimeError as exc:
            last = exc
            # 只有真人风控拒才继续降级。line_art 这一档以前是无条件往下掉的，
            # 于是参数类错误（例如注释里举的「参考视频短于 1.8s」）也会被当成风控，
            # 悄悄掉到 t2v、商品信息全丢，日志上只看到「降级成功」。
            if not is_real_person_reject(exc):
                raise
    raise RuntimeError("gen_video_safe 全部回退失败: %s" % last)
