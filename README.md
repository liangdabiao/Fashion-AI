# Fashion AI Promo Photo Pipeline

电商 AI 生图爆款流水线 — 输入新品平铺图，自动检索相似爆款、分析风格，生成专业宣传图。

## 项目简介

跨境电商场景中，商家每出一款新品都需要制作宣传图（模特穿着商品的照片）。传统方式需要请模特、搭场景、拍摄修图，成本高、周期长。

本项目实现了一套全自动的 AI 生图流水线：

1. **以图搜爆款** — 将新品平铺图与历史爆款库做混合检索（视觉相似 + 关键词匹配 + 销量过滤），找到风格最接近的爆款
2. **AI 分析风格** — 将爆款参考图发给大语言模型，自动分析场景、灯光、姿势、氛围，生成风格描述 prompt
3. **AI 生成宣传图** — 将新品图 + 爆款参考 + 风格描述一起发给图像生成模型，输出专业宣传图


项目灵感来源于教程：Nano Banana2+ Milvus+ qwen3.5，打造电商生图爆款流水线
文章参考来源：https://mp.weixin.qq.com/s/itMVbtZDW0IzIBhp6e9OHw

### 为什么用向量数据库？

电商最核心的资产是大量经过市场验证的爆款图。模特表现力、场景构图、光影质感，都是花钱试出来的。与其每次从零写 prompt，不如直接从历史爆款中检索最相似的视觉特征，让 AI 继承"爆款基因"。这正是 Milvus 混合检索发挥作用的地方。

### 项目简单使用：

  使用方式：
  1. 将真实商品图片放入 images/（替换占位图），新品放入 new_products/
  2. python main.py setup — 建库 + 编码 + 入库
  3. python main.py search --new-id NEW001 — 测试检索
  4. python main.py generate --new-id NEW001 — 完整生图流水线


### 为什么用混合检索（Dense + Sparse + Scalar）？

单一检索方式有盲区：

| 检索方式 | 擅长 | 弱点 |
|---------|------|------|
| Dense 向量（图片 Embedding） | 视觉相似度（颜色、款式、风格） | 不擅长精确关键词匹配 |
| Sparse 向量（TF-IDF） | 关键词语义（"floral"、"midi"、"chiffon"） | 不理解视觉特征 |
| 标量筛选 | 品类过滤、销量排序 | 无法理解语义 |

三者结合 + RRF（Reciprocal Rank Fusion）重排序，检索精度远超单一方式。

## 技术架构

```
新品平铺图
    │
    ▼
llama-nemotron-embed-vl-1b-v2 (Embedding API)
    │
    ├── Dense 向量 (2048d)  ──────────────────┐
    │                                          ▼
    └── 文本 → TF-IDF → Sparse 向量 ──→  Milvus 混合检索
                                              │
                                   Dense (视觉相似)
                                +  Sparse (关键词匹配)
                                +  Scalar (品类 + 销量过滤)
                                +  RRF 重排序
                                              │
                                              ▼
                                    Top-K 相似爆款宣传图
                                              │
                                              ▼
                                   Qwen 3.5 (多模态 LLM)
                                   分析场景/灯光/姿势/氛围
                                   输出风格描述 prompt
                                              │
                                              ▼
                                   Gemini 3.1 Flash / Nano Banana 2
                                   (新品图 + 爆款参考 + 风格prompt)
                                              │
                                              ▼
                                    生成的宣传图 → output/
```

### 使用的模型

所有模型通过 [OpenRouter](https://openrouter.ai/) API 调用，无需本地 GPU。

| 模型 | 用途 | 说明 |
|------|------|------|
| `nvidia/llama-nemotron-embed-vl-1b-v2` | 图片/文本 Embedding | 免费，输出 2048 维向量，支持图文同空间 |
| `qwen/qwen3.5-397b-a17b` | 爆款风格分析 | 多模态，可同时理解图片和文本 |

**宣传图生成模型**（通过 `--model` 参数切换）：

| 别名 | 模型 | 说明 |
|------|------|------|
| `nano-banana`（默认） | `google/gemini-3.1-flash-image-preview` | 支持多参考图融合、多比例（含 8:1/1:8 超宽超高）、文字渲染，单张约 $0.067，无地区限制 |
| `gpt-image` | `openai/gpt-5.4-image-2` | OpenAI 最新生图模型，人像融合效果好，**需海外网络环境** |
| `gpt-image-pro` | `openai/gpt-5-image` | OpenAI 生图 Pro 版，质量更高，**需海外网络环境** |
| `gpt-image-mini` | `openai/gpt-5-image-mini` | OpenAI 生图轻量版，成本更低，**需海外网络环境** |

### Milvus 混合 Schema

```
Collection: fashion_products
├── id              INT64 (主键，自增)
├── product_id      VARCHAR  — 商品编号
├── category        VARCHAR  — 品类 (maxi_dress, midi_dress, casual_top)
├── color           VARCHAR  — 颜色
├── style           VARCHAR  — 风格 (bohemian, floral, bodycon, ...)
├── season          VARCHAR  — 季节 (spring, summer, autumn)
├── sales_count     INT64    — 销量（用于标量过滤）
├── description     VARCHAR  — 商品描述（用于 TF-IDF）
├── price           FLOAT    — 价格
├── dense_vector    FLOAT_VECTOR(2048) — 图片 Embedding
└── sparse_vector   SPARSE_FLOAT_VECTOR  — TF-IDF 稀疏向量
```

## 项目结构

```
cloth-milvus/
├── main.py              # CLI 入口 (setup / search / generate)
├── config.py            # 配置管理，从 .env 读取
├── utils.py             # 工具函数 (图片编解码)
├── embeddings.py        # Dense/Sparse 向量生成
├── milvus_store.py      # Milvus 建库、插入、混合检索
├── style_analyzer.py    # Qwen3.5 风格分析
├── image_generator.py   # Nano Banana 2 宣传图生成
├── data_setup.py        # 示例数据生成 (CSV + 占位图)
├── requirements.txt     # Python 依赖
├── .env                 # API Key 配置 (需自行填写)
│
├── images/              # 历史爆款商品图片
├── new_products/        # 新品平铺图
├── products.csv         # 历史商品元数据 (40条示例)
├── new_products.csv     # 新品元数据 (4条示例)
└── output/              # 生成的宣传图输出目录
```

## 快速开始

### 1. 环境准备

- Python 3.10+
- [Zilliz Cloud](https://cloud.zilliz.com/) 账号（免费 tier 即可）
- [OpenRouter](https://openrouter.ai/) 账号和 API Key

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 .env

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxx
MILVUS_HOST=https://xxx.gcp.zillizcloud.com
MILVUS_TOKEN=your-zilliz-api-key
COLLECTION_NAME=fashion_products
```

- **OPENROUTER_API_KEY** — 在 [openrouter.ai/keys](https://openrouter.ai/keys) 获取
- **MILVUS_HOST / MILVUS_TOKEN** — 在 Zilliz Cloud 控制台创建集群后获取

### 4. 准备数据

#### 方式一：使用示例数据（快速体验）

将商品图片放入对应目录：

```
images/
├── SKU001.jpg    # 历史爆款商品图
├── SKU002.jpg
├── ...
└── SKU040.jpg

new_products/
├── NEW001.jpg    # 新品平铺图
├── NEW002.jpg
├── NEW003.jpg
└── NEW004.jpg
```

编辑 `products.csv` 和 `new_products.csv` 填入你的真实商品信息。首次运行 `setup` 时程序会自动创建 CSV 模板和占位图。

#### 方式二：使用自己的数据

编辑 CSV 文件，字段格式：

**products.csv（历史爆款库）：**

```csv
product_id,image_path,category,color,style,season,sales_count,description,price
SKU001,SKU001.jpg,maxi_dress,pink,bohemian,summer,3200,Bohemian pink maxi dress with tassel tie detail,29.99
```

- `category` — 品类，自由填写，会作为检索过滤条件
- `sales_count` — 历史销量，默认只检索销量 > 1500 的
- `description` — 英文描述，用于 TF-IDF 关键词匹配

**new_products.csv（新品）：**

```csv
new_id,image_path,category,style,season,prompt_hint
NEW001,NEW001.jpg,midi_dress,casual,spring,Knit cardigan on a sunlit cafe terrace
```

- `prompt_hint` — 场景提示词，告诉 AI 你希望生成什么样的宣传场景

### 5. 运行

```bash
# Step 1: 建库入库（首次运行或数据更新时执行）
python main.py setup

# Step 2: 测试检索（可选，验证混合检索是否正常）
python main.py search --new-id NEW001

# Step 3: 完整流水线（检索 → 风格分析 → 生图）
# 默认使用 Nano Banana 2 生图
python main.py generate --new-id NEW001

# 使用 GPT-5.4 Image 2 生图
python main.py generate --new-id NEW001 --model gpt-image

# 使用 GPT-5 Image Pro 生图
python main.py generate --new-id NEW001 --model gpt-image-pro
```

生成的宣传图保存在 `output/` 目录。

## CLI 参数

### setup

初始化数据，生成向量，创建 Milvus Collection 并入库。

```bash
python main.py setup
```

无需额外参数。重复运行会重建 Collection。

### search

为新品检索相似爆款。

```bash
python main.py search [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--new-id` | 全部 | 指定新品 ID，如 `--new-id NEW001` |
| `--top-k` | 3 | 返回结果数量 |
| `--sales-threshold` | 1500 | 最低销量过滤门槛 |

### generate

完整流水线：混合检索 → 风格分析 → 宣传图生成。

```bash
python main.py generate [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--new-id` | 全部 | 指定处理某个新品 |
| `--top-k` | 3 | 参考爆款数量 |
| `--sales-threshold` | 1500 | 最低销量过滤门槛 |
| `--aspect-ratio` | `3:4` | 生成图宽高比（支持 `1:1`, `3:4`, `4:1`, `9:16` 等） |
| `--image-size` | `2K` | 生成图分辨率（`512px`, `1K`, `2K`, `4K`） |
| `--model` | `nano-banana` | 生图模型：`nano-banana`, `gpt-image`, `gpt-image-pro`, `gpt-image-mini` |

## 开发者指南

### 模块说明

| 文件 | 职责 |
|------|------|
| `config.py` | 所有配置集中管理，从 .env 读取敏感信息 |
| `embeddings.py` | 图片 → Dense 向量（OpenRouter API）、文本 → TF-IDF Sparse 向量 |
| `milvus_store.py` | Milvus Collection CRUD 和混合检索（Dense + Sparse + RRF） |
| `style_analyzer.py` | 调用 Qwen3.5 多模态 LLM 分析爆款图片风格 |
| `image_generator.py` | 调用 Gemini/Nano Banana 2 生成宣传图 |
| `utils.py` | 图片 base64 编解码、OpenRouter 响应解析 |

### 关键技术点

**1. 混合检索流程**

```python
# 两路并行检索
dense_req = AnnSearchRequest(data=[query_dense], anns_field="dense_vector", ...)
sparse_req = AnnSearchRequest(data=[query_sparse], anns_field="sparse_vector", ...)

# RRF 融合
results = client.hybrid_search(reqs=[dense_req, sparse_req], ranker=RRFRanker(k=60))
```

RRF（Reciprocal Rank Fusion）算法公式：`score(d) = Σ 1/(k + rank(d))`，将两路检索的排名融合为一个综合分数。

**2. 多生图模型支持**

所有生图模型统一通过 OpenRouter `/api/v1/chat/completions` 调用，格式一致：

```python
payload = {
    "model": model_id,  # 如 "google/gemini-3.1-flash-image-preview" 或 "openai/gpt-5.4-image-2"
    "messages": [{"role": "user", "content": [...]}],
    "modalities": ["image", "text"],  # 声明输出包含图片
}
# Gemini 系列额外支持 image_config 控制分辨率
if "google" in model_id:
    payload["image_config"] = {"aspect_ratio": "3:4", "image_size": "2K"}
# OpenAI 系列支持 aspect_ratio
elif "openai" in model_id:
    payload["image_config"] = {"aspect_ratio": "3:4"}
```

模型差异：

| 特性 | Nano Banana 2 (Gemini) | GPT Image 系列 |
|------|----------------------|---------------|
| 超宽/超高比例 | 支持 4:1, 1:4, 8:1, 1:8 | 标准比例 |
| 分辨率控制 | 支持 0.5K~4K | 自动 |
| 文字渲染 | 支持多语言、手写体 | 支持 |
| 参考图融合 | 支持 14 张（10对象+4角色） | 支持多张 |
| 人像融合 | 偶有服装贴合问题 | 更自然 |

通过 `--model` 参数自由切换，同一套代码适配两种生图引擎。

**3. Qwen 3.x Thinking 模式**

Qwen 3.x 默认启用 thinking，会将推理过程放在 `reasoning_content` 而非 `content`。本项目通过 `enable_thinking: False` 关闭该行为，确保直接返回结果。

### 自定义扩展

- **换模型** — 修改 `config.py` 中的模型名称，所有模型均走 OpenRouter，无需改动调用代码
- **新增品类** — 直接在 CSV 中添加新 category，程序自动适配
- **调整检索策略** — 修改 `milvus_store.py` 中的 `filter_expr`、`RRFRanker` 参数或 `limit` 值
- **批量处理** — `generate` 不带 `--new-id` 即可批量处理所有新品

## 成本参考

| 模型 | 单次调用成本 |
|------|------------|
| Embedding (nvidia) | 免费 |
| Qwen 3.5 | ~$0.001/次 |
| Gemini 3.1 Flash (Nano Banana 2) | ~$0.067/张 |

处理一个新品（1次检索 + 1次风格分析 + 1张生图）约 $0.07。

## 已知限制

- 占位图无法用于实际检索和生图，必须替换为真实商品照片
- 图片分辨率影响 Embedding 质量，建议商品图不低于 800x800
- Nano Banana 2 在服装与人体融合上偶尔不自然，可通过分步生图（先生成场景背景，再生成模特）或使用 Pro 版本改善
- Milvus Lite 不支持 Windows，本项目使用 Zilliz Cloud 替代

## 依赖

```
pymilvus>=2.4.0
openai>=1.0.0
requests>=2.31.0
pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
python-dotenv>=1.0.0
numpy>=1.24.0
```


## 特别感谢

linux.do社区佬友支持：  https://linux.do ，
感谢 佬友 trader 提供 api支持