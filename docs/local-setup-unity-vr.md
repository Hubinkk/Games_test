# 本地环境搭建：HTML 网页游戏 + Unity VR

适用：在自己的电脑上接着做"HTML 与 VR 网页游戏"。本机已有 **Unity 6000.4.4f1**（Unity 6.4）。

## 0. 本地聊天窗口（Claude Code 在本机运行）

云端会话连不到你的电脑，所以要在本机另开一个会话。两种方式任选：

- **Claude 桌面 App**：打开 Claude Code 标签，选择本仓库所在文件夹，新建会话
- **终端**：进入仓库文件夹后运行下面的命令。它会出现在手机或网页的 Claude Code App 里，可以继续用手机聊

```bash
claude remote-control
```

新会话开场可以直接说："读 `docs/handoff.md` 和 `docs/local-setup-unity-vr.md`，接着做 HTML 与 VR 网页游戏"。

## 1. 拉取仓库

```bash
git clone https://github.com/Hubinkk/Games_test.git
```

```bash
cd Games_test && git checkout claude/busy-ramanujan-4jvlik
```

## 2. HTML 网页游戏（不需要安装任何东西）

`AnShui.html`、`BodyMap.html`、`FigureAnatomy.html` 都是单文件，双击即可打开。
要在 VR 头显（Quest 浏览器）里测试 WebXR，页面必须通过 **https 或 localhost** 访问：

```bash
python -m http.server 8000
```

- 电脑浏览器：打开 `http://localhost:8000/FigureAnatomy.html`
- Quest 头显：用 USB 连接后执行下面的端口转发，再在头显浏览器打开同一个地址

```bash
adb reverse tcp:8000 tcp:8000
```

（`adb` 来自 Android Platform Tools；Unity 安装 Android 模块时会自带一份。）

## 3. Unity 版本：要不要升级？

**建议：先不升级，直接用 6000.4.4f1。**

- 6.4 是 Update 版本，XR 和 WebGL 功能都够用
- 想要长期稳定的话，**6.3 LTS** 支持到 2027 年 12 月
- Unity 已经有更新的 6.x Update 版本。只有遇到具体 bug 或者需要新功能时，才通过 Unity Hub 安装最新的 6.x，不要中途频繁换版本

## 4. Unity Hub 里需要加装的模块

在 Unity Hub → Installs → 6000.4.4f1 → Add modules 中勾选：

| 模块 | 用途 |
|---|---|
| **Web Build Support** | 导出成网页（WebGL / WebGPU），可以和现有 HTML 作品放在一起 |
| **Android Build Support**（含 OpenJDK、Android SDK & NDK） | 打包到 Quest 等一体机 |
| Windows Build Support (IL2CPP) | 可选，打包 PC VR |

## 5. 新建 VR 项目

1. Unity Hub → New project → 模板选 **VR**（或 Universal 3D/URP 再手动加包）
2. Package Manager 中确认安装了以下包：
   - `XR Plug-in Management`
   - `OpenXR Plugin`（Quest 再加 `Meta XR` / OpenXR Meta 功能组）
   - `XR Interaction Toolkit`（导入 Starter Assets 示例）
3. Project Settings → XR Plug-in Management：PC 和 Android 两个页签都勾选 **OpenXR**
4. Unity 官方的 Web 导出**不内置 WebXR**。要在浏览器里跑 VR，加装社区包 **WebXR Export**（De-Panther/unity-webxr-export，通过 OpenUPM 安装）
5. 新项目建议放在本仓库下的 `unity/` 文件夹，并加 Unity 的 `.gitignore`（忽略 `Library/`、`Temp/`、`Logs/`、`UserSettings/`、`Obj/`、`Build/`）

## 6. 两条路线怎么选

| 路线 | 优点 | 缺点 |
|---|---|---|
| **纯 HTML + three.js + WebXR** | 和现有 3 个网页一致，单文件、发链接就能玩 | 复杂物理和大型场景比较吃力 |
| **Unity → WebXR 导出** | 编辑器、物理、动画都更强，同一个项目还能出 Quest 原生版 | 包体大，首次加载慢；WebXR 依赖社区包 |

建议：先用 HTML + WebXR 把 `FigureAnatomy.html` 做成能在头显里看的版本（阶段 A，见 `docs/second-body.md`）；需要物理、交互和打包时再迁到 Unity。

## 7. 环境自检清单

- [ ] Unity Hub 中能看到 6000.4.4f1，并且已装 Web / Android 模块
- [ ] `git --version`、`python --version` 都能输出版本号
- [ ] （有头显时）`adb devices` 能看到设备，头显已开启开发者模式
- [ ] 本机 Claude Code 会话已经打开，并读过 `docs/handoff.md`
