# Gemini 视频理解链路实测记录

实测日期 2026-09-09。样本取自 `output/web/tasks/` 下的历史成片，全部为真实调用。
结论先行：**链路可用，音频已经能听，base64 容量远比预期宽松，唯一确定的问题是默认网关列表首位已死。**

## 一、链路是什么

| 项 | 值 | 出处 |
|---|---|---|
| 默认引擎 | `gemini`（环境变量 `AIGC_ENGINE`） | `config.py:54` |
| 模型 | `gemini-3.1-pro-preview` | `config.py:71` |
| 网关 | 9 个内网裸 IP:端口，无域名、节点会漂移 | `config.py:58-63` |
| 端点 | `/service/llm/openairr` | `config.py:69` |
| 协议 | Gemini 原生 `contents.parts`，非 OpenAI 兼容 | `aigc.py:137` |
| 鉴权 | body 里的 `channel`，**没有** Bearer header | `aigc.py:156` |
| 成功判据 | `status.code == 0`，光看 HTTP 200 不够 | `aigc.py:173` |
| 上传方式 | **不上传**，`open()` 本地文件直接 base64 内联 | `aigc.py:93-117` |
| 回落链路 | wenchain `/chat/completions` + `ali-qwen3.7-plus`，媒体传公网 URL | `aigc.py:65-79` |

只有回落到 wenchain 时，`_public_media()` 才会 `storage.upload()` 到 BOS 预签名 URL，
并按 `路径+size+mtime` 缓存，同一文件只上传一次。

## 二、网关连通性（逐个发一次纯文本请求）

| 网关 | 结果 | 握手耗时 |
|---|---|---|
| `10.252.161.175:8018` | **死**，ConnectionError 直接拒连 | 0.1s |
| `10.252.161.216:8534` | 活 | 3.0s |
| `10.252.161.216:8535` | 活 | 3.5s |
| `10.252.164.45:2540` | 活 | 3.0s |
| `10.252.164.176:8049` | 活 | 3.3s |
| `10.252.164.237:2076` | 活 | 3.1s |
| `10.252.165.27:2056` | 活 | 2.7s |
| `10.252.165.158:2044` | 活 | 2.7s |
| `10.252.165.158:8113` | 活 | 3.2s |

8 活 1 死，而**死的那个正好是列表第一个**。`vision_gemini()` 按列表顺序试
（`aigc.py:153`），所以现在每一次视频理解调用都要先撞一次失败再切下一个。
把它从 `GEMINI_DEFAULT_GATEWAYS` 里挪到末尾或删掉即可，零风险。

也可以用环境变量临时绕开，不动代码：

```bash
export GEMINI_GATEWAY_URL=http://10.252.161.216:8534,http://10.252.161.216:8535
```

## 三、base64 能吃多大

同一网关 `10.252.161.216:8534`，用 ffmpeg 定码率压出阶梯样本，逐级加压：

| 目标 | 实际文件 | 请求 body | 耗时 | 结果 |
|---|---|---|---|---|
| 5MB | 5.46MB | 7.3MB | 19s | OK |
| 10MB | 10.34MB | 13.8MB | 13s | OK |
| 15MB | 14.86MB | 19.8MB | 19s | OK |
| 18MB | 17.70MB | 23.6MB | 20s | OK |
| 20MB | 19.61MB | 26.2MB | 22s | OK |
| 25MB | 24.41MB | 32.5MB | 18s | OK |
| 30MB | 29.22MB | 39.0MB | 22s | OK |
| 40MB | 39.05MB | 52.1MB | 33s | OK |
| 60MB | 60.79MB | 81.1MB | 26s | OK |
| 80MB+ | 76.32MB（x264 到顶，压不上去了） | **101.8MB** | 35s | OK |

**测到 76MB 文件 / 101.8MB body 都没被拒，没找到上限。** 不是 20MB 上限
（Gemini 官方文档对 inline data 的说法在这个网关上不适用，网关内部大概另做了中转）。
耗时也没随体积爆炸，13s→35s 基本线性。
再往上压需要换更长的源片，暂时没测；对当前业务（成片 15-30s、几 MB）已经远远够用。

## 四、音频：base64 内联本来就带着音轨

样本 `20260904-173528-a411/render/with_audio.mp4`（4.57MB，15s，有 TTS 配音），
以 `mimeType: video/mp4` 内联，问「逐字写出音轨里的台词」：

| 用例 | 结果 |
|---|---|
| 带音轨 | 转写出 5 句，与 `script.json` 的 `台词` 逐字基本一致（「教里」听成「组里」） |
| **去音轨（`-an` 对照组）** | 只回「没有人声」 |
| 只问声音、不许描述画面 | 「无音乐，有两名男声对话。音效包含柴火噼啪声、明显的喝水吞咽声和户外鸟鸣」 |

对照组是关键：这条片带烧制字幕，如果模型是在读画面上的字，去掉音轨后照样能答出台词。
它答不出来，说明**真的在听音频**，而且能分辨非人声细节（吞咽声、鸟鸣、有没有音乐）。

所以不需要为「同时理解音频」做任何改造——现有链路已经在用了：
- `analyze_reference.py:292` 把整条视频丢给 Gemini，拆解 prompt 里就有
  `台词` / `音效音乐` / `有背景音乐` / `有人声口播` 四个字段（`analyze_reference.py:88-106`）
- `ref_audio.py:178` 另外单独发一次抽出来的音轨（`type: audio`）做音乐性分类

真要加强，改的是 prompt 而不是传输方式。

## 五、BOS URL 也能给 Gemini（但没必要换）

3.22MB 样本：

| 方式 | 耗时 |
|---|---|
| qwen + BOS URL（基线） | 4.1s |
| gemini + BOS URL（现有代码：`requests.get` 拉回本地再 base64） | 9.8s |
| gemini + `fileData.fileUri` 直接透传 | 17.6s |

透传能通，`status.code == 0`。因此 `aigc.py:8` 与 `aigc.py:143` 的注释
**「媒体必须 base64 内联，不走 URL 透传」是错的**——那是代码的选择，不是网关的限制。

但透传更慢（拉取由对端做，还多付一次 BOS 上传），而 base64 到 100MB body 都没问题，
**现阶段没有任何理由改**。留档是为了将来真遇到超大原片时知道有这条路。

## 六、顺带发现的两个坑

**qwen 有最小时长/尺寸门槛。** 33KB 的极短片被拒：
`InvalidParameter: The video file is too s...`，同一条片 gemini 正常理解。
隐患是 gemini 挂掉回落 qwen 时，极短片段会直接失败而不是降级成功。

**解释器必须用 `/root/miniconda3/bin/python`。** `baidubce` 与 `imageio_ffmpeg`
只装在 conda base；`/usr/bin/python3` 是 3.6.8 且缺这两个包。`start_web.sh` 写的是裸
`python3`，靠 PATH 解析，非交互 shell 里会解析到错的那个。
另外 `/root/miniconda3/envs/media/bin/ffmpeg` **没编 libx264**，
要转码得用 `imageio_ffmpeg.get_ffmpeg_exe()` 那个二进制。

## 七、怎么复测

上面每一项都是一次性脚本跑出来的，没有落成常驻测试。要复现，核心就三步：
逐个网关发纯文本请求测活；用 `aigc._gemini_part()` 造 `inlineData` 后阶梯加压；
拿一条有配音的成片做「带音轨 / `-an` 去音轨」对照。
判成功一律看 `status.code == 0`，别看 HTTP 码。
