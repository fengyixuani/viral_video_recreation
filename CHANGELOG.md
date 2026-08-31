Changelog
===
以下记录了项目中所有值得关注的变更内容，其格式基于[Keep a Changelog]。

本项目版本遵守[Semantic Versioning]和[PEP-440]。

[Unreleased]
---
### Added
- 基础层：配置加载（`config.py`）、模型网关调用（`llm.py`）、BOS 存储（`storage.py`）、
  ffmpeg 媒体工具（`media.py`）、任务目录（`task_store.py`）、流水线骨架（`pipeline.py`，
  含 ingest 步骤、阶段并行与断点续跑）
- CLI 子命令：`doctor` / `new` / `run` / `list`
- 单测覆盖配置、任务目录、媒体工具、流水线与 CLI
### Changed
- 移除脚手架的 hello world demo，`cmdline` 换成复刻任务入口

0.1.0 - 2026-08-31
---
### Added
- 创建项目


[Unreleased]: http://icode.baidu.com/repos/baidu/wk-strategy/viral_video_recreation/merge/0.1.0...master

[Keep a Changelog]: https://keepachangelog.com/zh-CN/1.0.0/
[Semantic Versioning]: https://semver.org/lang/zh-CN/
[PEP-440]: https://www.python.org/dev/peps/pep-0440/
