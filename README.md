# 关系查询溯源与反事实解释 API

客户端一次请求提交 **多张带主键的关系表快照** 和一个 **受限关系代数计划**，
服务执行查询，并为每条输出返回：

* **全部极小来源行见证集**（minimal witness sets / why-provenance）；
* **布尔溯源表达式**（DNF：见证内部为合取 `∧`，见证之间为析取 `∨`）；
* 可选地，对指定输出计算 **令其消失的全部最小源行删除集**，并对每个删除集
  **在删除后的快照上重算整条计划**，返回重算结果。

技术栈：Python 3.11 · FastAPI · SQLite（幂等响应持久化）· 无外部服务依赖。

---

## 1. 一键启动（Docker Compose）

```bash
docker compose up --build
```

启动后：

| 用途 | 地址 |
| --- | --- |
| HTTP API | http://localhost:8000 |
| 交互式文档（Swagger） | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/health |

幂等响应持久化在 Docker 卷 `provenance-data`（容器内 `/data/provenance.db`）。

### 本地直接运行（不使用 Docker）

```bash
python3 -m pip install -r requirements.txt
PROVENANCE_DB=./data/provenance.db \
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

运行测试：

```bash
python3 -m pip install pytest
python3 -m pytest -q
```

---

## 2. API

### `POST /query`

请求体（JSON）：

```jsonc
{
  "idempotency_key": "readme-demo-1",        // 必填，绑定本次规范化请求
  "tables": [ ... ],                         // 1..n 张表快照（见下）
  "plan": { ... },                           // 1 个关系代数计划
  "explain": { "index": 0 }                  // 可选：对某条输出做反事实解释
}
```

**表快照**

```jsonc
{
  "table": "orders",
  "primary_key": ["oid"],                    // 必填，非空；支持复合主键
  "columns": [                               // 可省略（从行推断）
    {"name": "oid", "type": "int"},          // 类型: int/float/string/bool/any
    {"name": "cid", "type": "int"},
    {"name": "amount", "type": "int"}
  ],
  "rows": [ {"oid": 1, "cid": 10, "amount": 50}, ... ]
}
```

**关系代数计划节点**

| 节点 | 形式 |
| --- | --- |
| 扫描 | `{"type":"scan","table":"orders"}` |
| 选择/过滤 | `{"type":"filter","input":<node>,"predicate":<谓词>}` |
| 投影 | `{"type":"project","input":<node>,"columns":["a",{"from":"b","as":"c"}]}` |
| 等值连接 | `{"type":"join","left":<node>,"right":<node>,"on":[{"left":"cid","right":"cid"}]}` |
| 分组聚合 | `{"type":"aggregate","input":<node>,"group_by":["cid"],"aggregates":[{"fn":"sum","column":"amount","as":"total"}]}` |

* 谓词树：叶子 `{"column":"amount","op":">=","value":40}`；
  算子 `= <> < <= > >=` 以及 `is_null` / `is_not_null`；
  布尔组合 `{"and":[...]}`、`{"or":[...]}`、`{"not":{...}}`。
  谓词按 SQL 三值逻辑求值：与 `NULL` 比较得到 UNKNOWN，不满足选择条件。
* 聚合函数：`count`（`column` 可写 `"*"` 或具体列，后者忽略 NULL）、
  `sum` / `min` / `max`（自动忽略 NULL；全 NULL 时 `sum/min/max` 为 `null`）。
* 等值连接的连接列按 USING 语义合并；其余右表列名冲突时自动限定为
  `right.<列>`（可用 `right_alias` 指定限定名），或在投影中重命名。
* 计划只能引用已提交的表与其字段，否则返回 `ILLEGAL_REFERENCE` / `UNKNOWN_TABLE`。

**反事实目标选择器**（`explain`，二选一）：

* `{"index": 0}` —— 输出按全部列全序排序后的稳定下标；
* `{"row": {...}}` —— 与输出列完全相等的一行。

### 响应约定

* 所有响应都是 **规范化（canonical）JSON**：对象键按码点排序、无多余空白、
  无尾随换行；成功响应附带 `X-Idempotent-Replay: true/false`。
* 幂等：`(idempotency_key, 规范化请求指纹)` 绑定。重复请求 **逐字节一致**；
  同一键但规范化计划/快照/目标不同 → `409 IDEMPOTENCY_KEY_CONFLICT`。
  JSON 中对象键的书写顺序不影响绑定（指纹基于规范化 JSON 计算）。

---

## 3. 实际请求 / 响应示例

### 请求：过滤 + 等值连接 + 投影，并解释输出行 0

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
  "idempotency_key": "readme-demo-1",
  "tables": [
    {"table": "orders", "primary_key": ["oid"],
     "columns": [{"name":"oid","type":"int"},{"name":"cid","type":"int"},{"name":"amount","type":"int"}],
     "rows": [{"oid":1,"cid":10,"amount":50},{"oid":2,"cid":10,"amount":30}]},
    {"table": "customers", "primary_key": ["cid"],
     "columns": [{"name":"cid","type":"int"},{"name":"name","type":"string"}],
     "rows": [{"cid":10,"name":"Alice"}]}
  ],
  "plan": {
    "type": "project",
    "columns": ["name", {"from":"oid","as":"order_id"}, "amount"],
    "input": {
      "type": "filter",
      "predicate": {"column":"amount","op":">=","value":40},
      "input": {
        "type": "join",
        "left": {"type":"scan","table":"orders"},
        "right": {"type":"scan","table":"customers"},
        "on": [{"left":"cid","right":"cid"}]
      }
    }
  },
  "explain": {"index": 0}
}'
```

实际响应（线上传输为单行紧凑 JSON；这里为可读性格式化）：

```json
{
  "status": "ok",
  "symbol_table": [
    {"symbol": "s1", "source": "customers[10]"},
    {"symbol": "s2", "source": "orders[1]"}
  ],
  "result": {
    "columns": ["name", "order_id", "amount"],
    "row_count": 1,
    "rows": [
      {
        "row": {"name": "Alice", "order_id": 1, "amount": 50},
        "witness_count": 1,
        "minimal_witnesses": [
          {
            "labels": ["customers[10]", "orders[1]"],
            "symbols": ["s1", "s2"],
            "conjunction": "s1 ∧ s2"
          }
        ],
        "provenance": {
          "dnf": "(s1 ∧ s2)",
          "dnf_structured": [["s1", "s2"]],
          "labels_structured": [["customers[10]", "orders[1]"]]
        }
      }
    ]
  },
  "counterfactual": {
    "target_index": 0,
    "target": {"name": "Alice", "order_id": 1, "amount": 50},
    "target_provenance": {
      "dnf": "(s1 ∧ s2)",
      "dnf_structured": [["s1", "s2"]],
      "labels_structured": [["customers[10]", "orders[1]"]]
    },
    "minimal_deletion_count": 2,
    "minimal_deletion_sets": [
      ["customers[10]"],
      ["orders[1]"]
    ],
    "scenarios": [
      {
        "delete": ["customers[10]"],
        "target_present": false,
        "result": {"columns": ["name", "order_id", "amount"], "row_count": 0, "rows": []}
      },
      {
        "delete": ["orders[1]"],
        "target_present": false,
        "result": {"columns": ["name", "order_id", "amount"], "row_count": 0, "rows": []}
      }
    ]
  }
}
```

解读：输出行 `(Alice,1,50)` 的唯一极小见证是 `orders[1] ∧ customers[10]`
（两条源行合取）；让它消失的最小删除集有两个：删客户 `customers[10]`
或删订单 `orders[1]`。两种删除都会在重算后使目标消失。

### 多见证（DNF 多析取项）与分组聚合

对分组聚合而言，输出行即“组”。组的存在性溯源是其成员派生的极小析取；
让组消失必须命中其所有极小见证。例如组 `cid=10` 含两条订单，溯源为
`s1 ∨ s2`，唯一的最小删除集是同时删除这两行：

```json
"target_provenance": {"dnf": "(s1) ∨ (s2)", ...},
"minimal_deletion_sets": [["orders[1]", "orders[2]"]]
```

### 幂等重放逐字节一致

```bash
curl -s ... -d @req.json > a.json
curl -s ... -d @req.json > b.json
cmp a.json b.json && echo "byte-identical"   # byte-identical
# 第二次响应头: X-Idempotent-Replay: true
```

---

## 4. 结构化错误

错误统一为 `4xx` + 规范化 JSON：

```json
{"error": {"code": "DUPLICATE_PRIMARY_KEY",
           "message": "Table 't': duplicate primary key [1]",
           "details": {"table": "t", "primary_key": [1]}}}
```

| code | 触发条件 |
| --- | --- |
| `MALFORMED_REQUEST` | JSON 非法、请求不符合 schema、行含未声明列、表重名等 |
| `MALFORMED_PLAN` | 节点类型未知、谓词结构非法、投影/聚合别名冲突等 |
| `UNKNOWN_TABLE` | 计划引用未提交的表 |
| `ILLEGAL_REFERENCE` | 计划/谓词/分组/聚合引用不存在的字段 |
| `DUPLICATE_PRIMARY_KEY` | 同一表快照内主键重复 |
| `NULL_PRIMARY_KEY` | 主键的任一分量为 null |
| `EMPTY_AGGREGATE` | 聚合输入为空（无法形成任何组；含标量聚合） |
| `TYPE_ERROR` | 声明类型不符、跨类型比较/连接、对非数值求 `sum`、对布尔值做有序比较 |
| `TARGET_NOT_FOUND` | `explain` 的 index 越界或 row 不匹配任何输出 |
| `IDEMPOTENCY_KEY_CONFLICT` | 幂等键已绑定到另一个规范化请求（HTTP 409） |

---

## 5. 溯源语义说明

* 来源行标签为 `表名[主键值,...]`，复合主键按顺序拼接。符号表把标签稳定映射为
  `s1, s2, ...`（按表名、主键全序排序）。
* 见证集经过包含极小化：投影去列、自连接等造成的冗余派生会被剔除。
* 选择、等值连接、投影以及 `count/sum/min/max` 分组聚合对源行集合都是单调的，
  因此“令输出消失的最小删除集”恰好等于该输出全部极小见证的 **极小命中集**
  （hypergraph transversal）。聚合输出按组键标识：删除后该组为空即视为消失；
  每个删除场景都重算整条计划，并显式给出 `target_present`。
