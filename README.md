# 金融事件智能分析系统
本文件夹包含可运行原型、项目自研源码、技术报告、数据说明、模型与算法说明、测试结果和案例展示。演示视频按要求单独提交，**不在本文件夹内**。
## 目录

```text
competition_submission/
├── start_demo.sh                  # 一键启动本地原型
├── source/
│   ├── event-intelligence-frontend/ # Web 页面、Python API、测试
│   ├── financial_event_project/     # Task1—Task4 算法源码、配置、文档
│   └── scripts/                     # 工作区辅助脚本
├── runtime_data/                    # 原型实际读取的 38 份冻结结果和指标
├── evidence/                        # Task1/Task2 历史正式报告摘录原件
├── 技术报告.md
├── 数据说明.md
├── 模型与算法说明.md
├── 测试结果与案例展示.md
└── 文件清单.sha256
```

`source/` 收录当前项目中的自研可读源码、配置和测试，共 968 个文件。原始 Excel 新闻语料、模型权重、缓存、日志、历史备份和第三方仓库未放入包内；它们不是**原型展示**的运行依赖。完整离线训练复现仍需要对应数据与权重，见[数据说明.md](./数据说明.md)和[模型与算法说明.md](./模型与算法说明.md)。

## 环境与依赖

推荐 Linux、Python 3.8+、现代 Chrome/Edge 浏览器。原型服务仅用 Python 标准库，无需 pip/npm 安装。Node.js 18+ 用于附带的前端单元测试。离线实验的基础依赖见 `source/financial_event_project/requirements.txt`，包括 pandas 和 pyarrow；具体模型脚本还需 PyTorch、Transformers、CUDA、模型权重等，依脚本配置安装。

## 本地部署与运行

在本目录执行：

```bash
bash start_demo.sh
```

默认监听 `127.0.0.1:8080`。浏览器打开 `http://127.0.0.1:8080/auth.html`，演示账号 `demo@finsight.cn`，密码 `123456`。该账号仅用于本地演示，注册账户保存在本地文件；不要把此服务直接当作公网生产服务。

端口或监听地址可指定：

```bash
PORT=18080 HOST=127.0.0.1 bash start_demo.sh
```

按 `Ctrl+C` 停止。`start_demo.sh` 自动设置冻结数据目录与 SQLite 工作台状态路径，无需修改源码中的绝对路径。也可直接启动 `source/event-intelligence-frontend/api_server.py`，但需把 `EVENT_INTELLIGENCE_PROJECT_ROOT` 指向本包的 `runtime_data/financial_event_project`。本包未提供公网访问链接，以本地部署为交付方式。

主要页面：概览、事件监测、可信评估、演化追踪、预警中心、可拖拽知识图谱、任务链路回放。浏览器点击“开始接收事件”后按页面流程查看冻结结果；它不会重新执行离线模型。

## 配置项

| 配置 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `127.0.0.1` | API 服务监听地址 |
| `PORT` | `8080` | API 服务监听端口 |
| `EVENT_INTELLIGENCE_PROJECT_ROOT` | 由脚本设置 | 冻结产物目录 |
| `EVENT_INTELLIGENCE_STATE_DB` | `.runtime/workflow.sqlite3` | 本地预警确认、规则和关注条件状态 |
| `EVENT_INTELLIGENCE_WORKSPACE_ROOT` | 自动推断 | 可选，工作区根目录 |
| `js/api.js` 的 `USE_MOCK` | `false` | 默认读取真实冻结结果；仅离线开发可切为 mock |
| `js/api.js` 的 `BASE_URL` | `/api/v1` | API 前缀 |

## 测试

在本目录执行：

```bash
cd source/event-intelligence-frontend
python3 -m unittest discover -s tests -v
node --test tests/mock-api.test.cjs
node --check js/app.js
node --check js/api.js
```

再启动原型，检查：

```bash
curl -f http://127.0.0.1:8080/api/v1/dashboard
curl -f http://127.0.0.1:8080/api/v1/events/EC-001740
```

浏览器自动化测试位于 `tests/workflow.spec.cjs`，需安装 `@playwright/test` 和 Chromium；运行前应使用独立的 `EVENT_INTELLIGENCE_STATE_DB`，避免测试修改演示状态。详细结果与 16 条覆盖正常、异常、边界场景的用例见[测试结果与案例展示.md](./测试结果与案例展示.md)。

## 常见问题

| 现象 | 处理 |
|---|---|
| 8080 端口被占用 | `PORT=18080 bash start_demo.sh` |
| 页面无事件 | 确认 `runtime_data/financial_event_project/outputs/event_credibility/event_credibility_scores_fulltext.jsonl` 存在，并通过脚本启动 |
| 登录失败 | 使用演示账号，注意密码为 `123456`；查看浏览器控制台和服务终端 |
| 图谱不显示 | 查看 `/api/v1/knowledge-graph?scope=company` 是否返回 `nodes` 和 `links` |
| 工作台数据未保存 | 检查 `.runtime/` 目录写权限及 `EVENT_INTELLIGENCE_STATE_DB` |
| 离线模型脚本缺依赖/权重 | 本包保证原型运行，不包含原始语料、第三方仓库和大模型权重；按对应脚本与配置补齐 |
