# Chatbot 架构文档

## 总体架构

```
用户 → Gradio UI → HTTP/SSE → FastAPI → ChatEngine → LangChain → LLM API
                                    ↓
                              SessionManager
                                    ↓
                              StorageBackend → SQLite
```

## 分层设计

### 1. 表示层 (src/ui/web/)
- Gradio Web UI
- 负责用户交互、消息展示、会话管理界面

### 2. 接口层 (src/interface/)
- ui_protocol.py: 定义 UI 与后端的通信协议

### 3. 应用层 (src/main.py + src/core/)
- main.py: FastAPI 路由定义
- chat_engine.py: 对话引擎，LLM 调用封装
- session_manager.py: 会话生命周期管理 + 内存缓存
- user_manager.py: 用户管理
- preset_manager.py: 角色预设管理
- config_manager.py: 统一配置管理

### 4. 领域层 (src/models/)
- schemas.py: Pydantic 数据模型（请求/响应/实体）

### 5. 基础设施层 (src/storage/)
- base.py: StorageBackend 抽象基类
- sqlite_backend.py: SQLite 实现（含 ORM 模型）
- factory.py: 工厂函数

## 数据流

### 流式聊天
1. 用户在 Gradio 输入消息
2. UI 通过 httpx 发送 POST /chat/stream（SSE）
3. FastAPI 调用 ChatEngine.chat_stream()
4. ChatEngine 从 SessionManager 获取历史记忆
5. ChatEngine 从 PresetManager 获取系统提示词
6. ChatEngine 构建消息链，调用 LLM.astream()
7. 逐 token 通过 SSE 返回 UI
8. 对话结束后保存上下文到内存和数据库

### 角色切换
1. 用户在 UI 选择新角色
2. UI 调用 PUT /sessions/{id}/role
3. 后端更新会话角色记录
4. 后端从数据库加载该角色的独立消息历史
5. UI 刷新聊天窗口，显示新角色的历史记录
