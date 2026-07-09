# langchain-chat

基于 LangChain 的多轮会话系统，支持多用户、多模型切换、流式输出、会话管理和角色切换。采用分层架构设计，全链路异步，具备可插拔存储能力。

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 核心框架 | LangChain 1.3.x |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | Gradio |
| 数据库 | SQLite（异步驱动 aiosqlite） |
| 异步模式 | 全链路 asyncio |
| 流式输出 | Server-Sent Events (SSE) |
| 配置管理 | PyYAML + python-dotenv |

## 主要功能

- **多轮对话** — 基于 LangChain `ConversationBufferMemory`，保持完整上下文连贯性
- **流式输出** — Server-Sent Events 逐 token 实时推送，无需等待完整响应
- **多用户支持** — 用户创建与切换，会话数据完全隔离
- **会话管理** — 新建、列表、切换、重命名、删除；自动保存；首条消息自动生成标题
- **角色切换** — 内置 5 种角色预设，切换时加载对应的独立对话历史
- **多模型支持** — 支持 DeepSeek、通义千问、智谱 GLM、Kimi 等 OpenAI 兼容接口
- **可插拔存储** — 抽象存储层，当前支持 SQLite，可扩展至 MySQL / PostgreSQL
- **统一配置** — `.env` + `config.yaml` + `logging.yaml` 三层配置分离

## 项目结构

```
langchain-chat/
├── .env                        # 敏感配置（API Key 等，不入库）
├── .env.example                # 环境变量模板
├── config.yaml                 # 全局配置
├── config/
│   ├── presets.yaml            # 角色预设定义
│   └── logging.yaml            # 日志配置
├── data/
│   └── sqlite/                 # SQLite 数据库文件
├── src/
│   ├── __init__.py
│   ├── main.py                 # 统一入口（FastAPI + Gradio）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chat_engine.py      # 对话引擎（LLM 调用 + 流式/非流式）
│   │   ├── session_manager.py  # 会话管理 + 内存缓存
│   │   ├── user_manager.py     # 用户管理
│   │   ├── preset_manager.py   # 角色预设管理
│   │   └── config_manager.py   # 统一配置管理
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic 数据模型
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py             # 存储抽象基类
│   │   ├── factory.py          # 存储工厂函数
│   │   └── sqlite_backend.py   # SQLite 实现（ORM + CRUD + 迁移）
│   ├── interface/
│   │   ├── __init__.py
│   │   └── ui_protocol.py      # UI 通信协议定义
│   └── ui/
│       └── web/
│           ├── __init__.py
│           └── app.py          # Gradio Web UI
├── scripts/
│   └── init_db.py              # 数据库初始化脚本
├── tests/
│   └── __init__.py
├── docs/
│   └── architecture.md         # 架构说明文档
├── pyproject.toml              # 项目元数据
└── requirements.txt            # 依赖清单
```

### 架构分层

```
用户  →  Gradio UI  →  HTTP/SSE  →  FastAPI  →  ChatEngine  →  LangChain  →  LLM API
                                            ↓
                                      SessionManager
                                            ↓
                                      StorageBackend  →  SQLite
```

## 快速开始

### 环境准备

- Python 3.10 或更高版本
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖

### 1. 克隆项目

```bash
git clone https://github.com/himawari95/chatbot.git
cd chatbot
```

### 2. 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 uv（更快）
uv pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，至少填入 DeepSeek API Key
# 其他模型的 Key 为可选项
```

`.env` 最小配置：

```ini
DEEPSEEK_API_KEY=你的DeepSeek_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEFAULT_MODEL=deepseek-chat
```

### 4. 初始化数据库

```bash
python scripts/init_db.py
```

### 5. 启动项目

```bash
# 同时启动后端 API 和 Web UI
python -m src.main
```

启动后访问：

- **Web UI**: http://127.0.0.1:7860
- **API 文档 (Swagger)**: http://127.0.0.1:8000/docs
- **API 文档 (ReDoc)**: http://127.0.0.1:8000/redoc

## 配置说明

### .env — 敏感配置

```ini
# DeepSeek（默认）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 通义千问（备用）
DASHSCOPE_API_KEY=你的Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 智谱 GLM（备用）
ZHIPU_API_KEY=你的Key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# Kimi（备用）
MOONSHOT_API_KEY=你的Key
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1

# 默认模型
DEFAULT_MODEL=deepseek-chat
```

### config.yaml — 全局配置

```yaml
server:
  host: "127.0.0.1"
  port: 8000

database:
  type: "sqlite"               # 存储后端类型
  path: "data/sqlite/chatbot.db"

llm:
  default_model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 2048

ui:
  host: "127.0.0.1"
  port: 7860
  share: false
```

### 角色预设

| 角色 | 标识 | 说明 |
|------|------|------|
| 🤖 默认 | `default` | 友好的 AI 助手 |
| 👨‍🏫 老师 | `teacher` | 耐心讲解，多用例子 |
| 👨‍💻 程序员 | `programmer` | 简洁技术向，给出完整代码 |
| 🧠 哲学家 | `philosopher` | 深度思考，引导思考本质 |
| 🤝 朋友 | `friend` | 轻松幽默，像朋友聊天 |

可在 `config/presets.yaml` 中自定义角色。

## 使用说明

### 登录

在页面顶部输入用户名，点击「登录 / 切换」。首次登录自动创建账号。

### 发送消息

在底部输入框输入消息，点击「发送」或按 Enter。AI 回复将逐字流式输出。

### 切换角色

在「🎭 角色」下拉框中选择不同人设。每个角色拥有独立的对话历史，切换后加载对应历史。

### 会话管理

| 操作 | 方式 |
|------|------|
| 新建会话 | 点击「＋ 新会话」 |
| 切换会话 | 在「会话」下拉框中选择 |
| 重命名 | 输入新标题后点击「✏️ 重命名」 |
| 删除会话 | 选中后点击「🗑」 |

### 切换模型

在「模型」下拉框中选择已配置的模型。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（含数据库状态） |
| `GET` | `/models` | 可用模型列表 |
| `GET` | `/presets` | 角色预设列表 |
| `POST` | `/users/login` | 用户登录/注册 |
| `POST` | `/chat` | 发送消息（完整响应） |
| `POST` | `/chat/stream` | 发送消息（SSE 流式） |
| `GET` | `/sessions` | 用户会话列表 |
| `POST` | `/sessions` | 创建新会话 |
| `DELETE` | `/sessions/{id}` | 删除会话 |
| `PUT` | `/sessions/{id}` | 重命名会话 |
| `PUT` | `/sessions/{id}/role` | 切换会话角色 |
| `GET` | `/sessions/{id}/messages` | 获取消息历史 |

完整 API 文档请访问 http://127.0.0.1:8000/docs

## 常见问题

### Q: 启动时提示 `ModuleNotFoundError`

运行 `pip install -r requirements.txt` 安装所有依赖。注意需要同时安装 `langchain-classic` 包。

### Q: 发送消息无响应

检查 `.env` 中的 API Key 是否正确，以及后端是否正常启动（终端应有 `Uvicorn running on...` 日志）。

### Q: 如何添加新模型

1. 在 `.env` 中添加对应模型的 `API_KEY` 和 `BASE_URL`
2. 在 `config.yaml` 的 `models` 列表中添加新条目

### Q: 如何自定义角色

编辑 `config/presets.yaml`，按已有格式添加新角色即可。

### Q: 能否使用其他数据库

存储层采用可插拔设计，只需实现 `src/storage/base.py` 中的 `StorageBackend` 抽象类，然后在 `src/storage/factory.py` 中注册即可。

## 许可证

MIT License

Copyright (c) 2025
