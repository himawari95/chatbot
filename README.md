# 🤖 AI 智能聊天机器人（LangChain Chat）

> 基于 LangChain 的多轮会话系统，支持多用户、多模型切换、流式输出、会话管理与角色切换。

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square)

## 📸 界面预览

> 截图待补充：请将图片放入 `screenshots/` 目录，并替换下方路径。

| 对话界面 | 角色切换 | 会话管理 |
|:---:|:---:|:---:|
| ![对话界面](screenshots/chat.png) | ![角色切换](screenshots/roles.png) | ![会话管理](screenshots/sessions.png) |

## ✨ 主要功能

- **多轮对话** — 基于 LangChain `ConversationBufferMemory`，保持完整上下文连贯性
- **流式输出** — Server-Sent Events 逐 token 实时推送
- **多用户支持** — 用户创建与切换，会话数据完全隔离
- **会话管理** — 新建、列表、切换、重命名、删除；首条消息自动生成标题
- **角色切换** — 内置 5 种角色预设，各自维护独立对话历史
- **多模型支持** — DeepSeek、通义千问、智谱 GLM、Kimi 等 OpenAI 兼容接口
- **可插拔存储** — 抽象存储层，当前支持 SQLite，可扩展至 MySQL / PostgreSQL

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 核心框架 | LangChain 1.3.x |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | Gradio |
| 数据库 | SQLite（异步驱动 aiosqlite） |
| 流式输出 | Server-Sent Events (SSE) |
| 配置管理 | PyYAML + python-dotenv |

## 🏗 架构分层

```
用户 → Gradio UI → HTTP/SSE → FastAPI → ChatEngine → LangChain → LLM API
                                        ↓
                                  SessionManager
                                        ↓
                                  StorageBackend → SQLite
```

## 🚀 运行方式

```bash
git clone https://github.com/himawari95/chatbot.git
cd chatbot
pip install -r requirements.txt

cp .env.example .env   # 填入 DEEPSEEK_API_KEY 等
python scripts/init_db.py
python -m src.main
```

启动后访问：

- **Web UI**: http://127.0.0.1:7860
- **API 文档 (Swagger)**: http://127.0.0.1:8000/docs

## 📄 License

MIT License

## 👤 作者

[himawari95](https://github.com/himawari95)
