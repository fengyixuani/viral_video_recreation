# 系统性修正：花字回归参考片驱动 + 重跑产物历史留档（用后即删）
import io

def patch(path, pairs):
    src = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert src.count(old) == 1, "NOT UNIQUE in %s: %r" % (path, old[:60])
        src = src.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(src)
    print("patched", path, "(%d)" % len(pairs))

# 1) write_script：花字不再跟「无口播」绑定，只跟参考片有无字幕/花字绑定
patch("write_script.py", [
    ("""硬性要求：
- 花字：参考片没有口播（全片台词为空）时，每个信息点镜头必须给「花字」短文案（14字内，
  传达该镜的卖点/信息），观众静音刷也能看懂；有台词的镜「花字」留空，不要双行叠字。
""",
     """硬性要求：
- 花字：只有参考片分镜的「特效」里出现字幕/花字/文案类元素时才写「花字」（14字内，
  承接参考镜文案的信息功能）；参考片画面干净无字的，所有镜的「花字」留空——
  成片字幕跟随参考片，参考片没有口播也没有花字，成片就不该有任何叠加文字。
"""),
    ('''    keep = ("序号", "时长秒", "景别", "运镜", "转场", "叙事功能", "画面", "动作", "台词")''',
     '''    keep = ("序号", "时长秒", "景别", "运镜", "转场", "叙事功能", "画面", "动作", "台词",
            "特效")   # 特效里有没有字幕/花字决定新剧本要不要写「花字」'''),
])

# 2) pipeline.reset_from：重置步骤的产物目录挪进 history/ 留档，重跑不覆盖旧结果
patch("pipeline.py", [
    ('''def reset_from(task_id: str, step: str) -> dict:
    """从某个步骤开始重跑：清掉它及其后续步骤的状态，产物留着当备份。"""
    rec = load(task_id)
    names = [n for n, _ in STEPS]
    if step not in names:
        raise ValueError("未知步骤：%s" % step)
    for name in names[names.index(step):]:
        rec["steps"].pop(name, None)
    rec.update({"status": "draft", "step": "", "error": ""})
    rec.pop("result", None)
    log(rec, "已重置步骤 %s 及其后续" % STEP_TITLES[step])
    return save(rec)''',
     '''# 每步的主产物目录：reset_from 重跑前把这些目录挪进 history/ 留档，
# 新一轮从零写，旧结果永远可回看（用户要求：重新生成不覆盖之前的结果）
STEP_DIRS = {"reference": ["reference"], "audio": ["audio"], "materials": ["materials"],
             "product": ["product"], "script": ["script", "assets"], "match": ["edit"],
             "generate": ["generated"], "compose": ["render"]}


def reset_from(task_id: str, step: str) -> dict:
    """从某个步骤开始重跑：清掉它及其后续步骤的状态，旧产物挪进 history/ 留档。"""
    rec = load(task_id)
    names = [n for n, _ in STEPS]
    if step not in names:
        raise ValueError("未知步骤：%s" % step)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kept = []
    for name in names[names.index(step):]:
        rec["steps"].pop(name, None)
        for sub in STEP_DIRS.get(name, ()):
            src = _d(task_id, sub)
            if os.path.isdir(src) and os.listdir(src):
                dst = os.path.join(task_dir(task_id), "history", stamp, sub)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
                kept.append(sub)
    report = os.path.join(task_dir(task_id), "report.md")
    if kept and os.path.isfile(report):     # 旧报告跟着旧产物一起留档
        os.rename(report, os.path.join(task_dir(task_id), "history", stamp, "report.md"))
    rec.update({"status": "draft", "step": "", "error": ""})
    rec.pop("result", None)
    log(rec, "已重置步骤 %s 及其后续%s" % (STEP_TITLES[step],
        "，旧产物留档 history/%s/（%s）" % (stamp, "、".join(kept)) if kept else ""))
    return save(rec)'''),
])
print("DONE")
