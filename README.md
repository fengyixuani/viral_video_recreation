Viral Video Recreation
===
通用爆款复刻。当前仓库只含基础层：配置、模型网关调用、BOS 存储、ffmpeg 媒体工具、
任务目录与流水线骨架。参考片拆解、素材切片、剧本仿写、分段出片等能力后续按步骤接入。

快速开始
---
```bash
pip3 install -r requirements.txt        # 装依赖（含本包 -e .）

# 密钥与内网网关地址写进本地私密配置（不入库）
cp viral_video_recreation/conf/config.env viral_video_recreation/conf/config.local.env
vi viral_video_recreation/conf/config.local.env   # 填 WENCHAIN_API_KEY / BOS_* / GEMINI_GATEWAY_URL

python3 -m viral_video_recreation doctor          # 环境自检：配置、ffmpeg、画布
python3 -m viral_video_recreation new --ref 参考片.mp4 --image 商品图.jpg --name 商品名
python3 -m viral_video_recreation run <task_id>   # 已完成的步骤自动跳过
python3 -m viral_video_recreation list
```

产物默认落在当前目录的 `output/`，可用 `VVR_OUTPUT_DIR` 改。一个任务 =
`output/tasks/{task_id}/`，参数、中间产物、状态、日志全在 `task.json` 里。

模块
---
* `config.py`：配置加载，优先级 环境变量 > `conf/config.local.env` > `conf/config.env`
* `llm.py`：模型调用。`chat` / `vision`（wenchain OpenAI 兼容）、`vision_gemini`
  （Gemini 原生协议，媒体 base64 内联）、`understand`（默认 gemini，不通回落 wenchain）、
  `gen_image` / `gen_video`
* `storage.py`：BOS 上传（本地文件 → 公网预签名 URL）与结果下载
* `media.py`：ffmpeg 薄封装。探时长/有无音轨、裁片段（短素材放慢补时长）、抽音轨、抽帧、
  归一化、concat
* `task_store.py`：任务目录与素材登记
* `pipeline.py`：步骤编排、阶段并行、断点续跑、`reset_from` 旧产物留档

测试
---
```bash
python3 -m pytest viral_video_recreation/tests -q
```
用例在本地维护、不入库（见 `.gitignore`）。媒体用例会真跑 ffmpeg（临时目录里生成 1 秒
测试片），不依赖网络与密钥。

如何贡献
---
贡献patch流程及质量要求

版本信息
---
本项目的各版本信息和变更历史可以在[这里][changelog]查看。

维护者
---
### owners
* fangmuyuan(fangmuyuan@baidu.com)

### committers
* fangmuyuan(fangmuyuan@baidu.com)

讨论
---
百度Hi交流群：群号


[changelog]: http://icode.baidu.com/repos/baidu/wk-strategy/viral_video_recreation/blob/master:CHANGELOG.md
