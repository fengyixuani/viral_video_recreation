"""54f7 从 seedance 生成步骤重跑（step_generate → step_compose）。用后即删。"""

# 从仓库根目录跑：python experiments/_regen_54f7.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pipeline

TID = "20260814-132850-54f7"

rec = pipeline.load(TID)
pipeline.step_generate(rec)
pipeline.save(rec)
print("GENERATE_DONE", flush=True)
pipeline.step_compose(rec)
pipeline.save(rec)
print("REGEN_54F7_DONE", flush=True)
