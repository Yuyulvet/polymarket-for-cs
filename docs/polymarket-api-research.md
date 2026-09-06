# Polymarket 调研报告（面向 CS2 自动交易程序）

> 调研日期：2026-08-22。来源：docs.polymarket.com 官方文档 + 对生产 API 的实时验证。
> 注意：官方文档已重构，旧路径 `/developers/CLOB/*`、`/developers/gamma/*` 大量 404。全站索引：https://docs.polymarket.com/llms.txt

## 1. 三大 API 与 Base URL

| API | Base URL | 用途 | 认证 |
|---|---|---|---|
| Gamma API | `https://gamma-api.polymarket.com` | 市场/赛事元数据、搜索、行情快照 | 读取免认证 |
| CLOB API | `https://clob.polymarket.com` | 订单簿、价格、下单撤单、价格历史 | 读免认证；写需 L1/L2 签名 |
| Data API | `https://data-api.polymarket.com` | 持仓、成交、持有者 | 读取免认证 |

链：**Polygon (chainId 137)**。抵押品：**pUSD**（已取代 USDC，6 位小数整数编码）。

关键合约：
- CTF Exchange（标准市场）：`0xE111180000d2663C0091e4f400237545B87B996B`
- Neg Risk CTF Exchange：`0xe2222d279d744050d28e00520010520000310F59`
- Conditional Tokens (CTF)：`0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`

## 2. 市场机制

- 每个市场 = 二元问题，Yes/No 两个 ERC1155 代币，每对由 $1 pUSD 全额抵押。
- **价格 $0.00–$1.00 = 隐含概率**。获胜代币结算后赎回 $1/枚，失败归零。
- 四操作：Split（$1 → 1 Yes + 1 No）/ Trade / Merge（1 Yes + 1 No → $1）/ Redeem。
- 生命周期：`live → matched → (delayed/unmatched) → 成交结算 MATCHED→MINED→CONFIRMED`。

### 订单类型（CLOB，链下撮合 + 链上结算）
- GTC（撤前有效）、GTD（到期自动过期；实际提前 1 分钟过期，到期须 ≥3 分钟后）
- FOK（全成或全撤）、FAK（市价默认，能成多少成多少）；市价单 `expiration="0"` 永不挂簿
- Post-only（会立即成交则拒绝，保证 maker）

### Tick size 与精度（市场可动态变更，需监听 `tick_size_change`）
| Tick | 价格小数 | 数量小数 | 金额小数 |
|---|---|---|---|
| 0.1 | 1 | 2 | 3 |
| 0.01 | 2 | 2 | 4 |
| 0.005 | 3 | 2 | 5 |
| 0.0025 | 4 | 2 | 6 |
| 0.001 | 3 | 2 | 5 |
| 0.0001 | 4 | 2 | 6 |

最小下单量：订单簿 `min_order_size` 字段（常见 5 股）。批量下单 1–15 笔/请求。

### negRisk 多结果市场
- 多结果事件（如锦标赛冠军）= 多个二元市场 + negRisk 组，恰好一个 Yes。
- 1 股 No 可经 Neg Risk Adapter 原子转换为其他每个市场各 1 股 Yes（套利工具）。
- CS2 冠军盘实测 `neg_risk: true`，交易时使用 negRisk 交易所合约签名。

## 3. 费用（对策略至关重要）

- **Maker 免费且有返佣；只有 Taker 付费。** fee = 股数 × feeRate × p × (1−p)，50¢ 时最高，越接近 0/1 越低。
- Sports/电竞类：taker 0.05，maker 返佣 15%。
- 例：100 股 @50¢ 的 sports taker 费 = $1.25。
- 持仓奖励：合格市场 4% 年化。

## 4. 体育/CS2 市场特点

- Tag 实测：**Esports id=64 (slug `esports`)，CS2 id=100677 (slug `cs2`)**。
- 发现方式：`GET /markets?tag_id=100677&closed=false&order=volumeNum&ascending=false`；或 `GET /events?tag_slug=cs2`；或全文搜索 `GET /public-search?q=队名`。
- CS2 冠军盘等不符合传统 sports 三向结构（`sportsMarketType/gameId` 为 null），主要靠 tag + 文本匹配。
- slug 规律：`will-team-spirit-win-the-cs2-ewc-2026-<时间戳>` / event `ewc-2026-cs2-winner-<时间戳>`。
- **开赛时刻自动撤销所有未成交限价单**（清簿）。
- **比赛期间有撮合延迟窗口**（`secondsDelay`，延迟中的订单不可撤销）——影响赛中策略。
- 结算走 UMA 乐观预言机：无争议约 2 小时；有争议 4–6 天。
- 实时比分流（公开 WS）：`wss://sports-api.polymarket.com/ws`，服务端 5s 发 ping，客户端 10s 内须回 pong。字段：slug、live、ended、score、period、elapsed、finished_timestamp。CS2 未在 period 表单列但同一基础设施覆盖。

## 5. 认证与签名

### L1（仅用于创建/派生 API 凭据）
- 头：`POLY_ADDRESS` / `POLY_SIGNATURE` / `POLY_TIMESTAMP` / `POLY_NONCE`
- EIP-712，domain `{name:"ClobAuthDomain", version:"1", chainId:137}`，message 固定 `"This message attests that I control the given wallet"`
- `POST /auth/api-key`（创建）/ `GET /auth/derive-api-key`（派生）→ `{apiKey, secret, passphrase}`

### L2（所有私有交易请求）
- 头：`POLY_ADDRESS` / `POLY_API_KEY` / `POLY_PASSPHRASE` / `POLY_TIMESTAMP` / `POLY_SIGNATURE`
- 签名：`urlsafeBase64( HMAC-SHA256( base64decode(secret), timestamp + METHOD + path + body原文 ) )`
- **query 参数不参与签名**；body 序列化必须与实际发送字节完全一致。

### signature_type（funder/signer 配错 = 静默失败，最常见坑）
| type | 钱包 | maker(funder) | signer | 适用 |
|---|---|---|---|---|
| 0 | EOA | EOA 地址 | 同一地址 | 链上直接交易（需 allowlist） |
| 1 | Proxy | 代理钱包地址 | 账户 signer | **网页邮箱/Magic/Google 注册** |
| 2 | Safe | Safe 地址 | 外部 signer | **MetaMask 等外部钱包连接创建** |
| 3 | Deposit Wallet | DW 地址 | 同 DW（ERC-7739） | **2026-05-04 后新账户默认** |

### 下单 EIP-712
domain `{name:"Polymarket CTF Exchange", version:"2", chainId:137, verifyingContract: 按 neg_risk 选择}`；Order struct：`salt, maker, signer, tokenId, makerAmount, takerAmount, side(0/1), signatureType, timestamp(ms), metadata, builder`。

## 6. 关键端点速查

### 行情（公开）
- `GET /book?token_id=`（bids/asks、min_order_size、tick_size、neg_risk）
- `POST /books`（≤500）、`GET|POST /price(s)`、`/midpoint(s)`、`/spread(s)`、`/last-trade-price(s)`
- `GET /prices-history?market=<token_id>&interval=1h|6h|1d|1w|max&fidelity=<分钟>&startTs=&endTs=` → `{history:[{t,p}]}`
- `GET /sampling-markets?next_cursor=`（全量市场游标翻页）

### 交易（L2）
- `POST /order`、`POST /orders`（1–15 笔）
- `GET /data/order/{id}`、`GET /data/orders`、`GET /data/trades`
- `DELETE /order`（body {orderID}）、`DELETE /orders`（1–3000）、`DELETE /cancel-market-orders`、`DELETE /cancel-all`
- `POST /v1/heartbeats`（心跳保单：每 5s 重发，10s 无心跳自动全撤——安全网机制）
- 余额/授权缓存：`GET /balance-allowance`、`GET /balance-allowance/update`（卖某 token 前首次需刷新 CONDITIONAL）

### Gamma / Data
- `GET /markets?tag_id=100677&closed=false&order=volumeNum&ascending=false&limit=&offset=`
- `GET /events?tag_slug=cs2`、`GET /public-search?q=`、`GET /markets/by-token/{token_id}`
- `GET /positions?user=`、`GET /trades?user=`、`GET /holders?market=`

### Gamma 字段注意
`outcomes` / `outcomePrices` / `clobTokenIds` 都是**字符串化 JSON 数组**，需二次 `json.loads`；clobTokenIds[i] 对应 outcomes[i]。关键字段：conditionId、bestBid/bestAsk/spread/lastTradePrice、orderPriceMinTickSize、orderMinSize、volume*、liquidity*、negRisk（event 层）。

## 7. SDK（重要：旧 SDK 已死）

- **`py-clob-client` 已于 2026-05-25 归档，官方声明 "no longer functional"**。网上大量教程/开源 bot 均过时。
- 新统一 SDK：Python **`polymarket-client`**（GitHub `Polymarket/py-sdk`），TS `@polymarket/client`。
- 新 SDK 自动处理：L1/L2 认证、tick size、negRisk 合约选择、精度取整。

```python
from polymarket import PublicClient, SecureClient

with PublicClient() as c:
    book = c.get_order_book(token_id=...)

with SecureClient.create(private_key=PK) as c:
    resp = c.place_limit_order(token_id=tid, side="BUY", price="0.52", size="10")
    c.cancel_order(order_id=...); c.cancel_all()
    est = c.estimate_market_price(token_id=tid, side="SELL", shares="10", order_type="FAK")
    c.wait_for_order_fill_settlement(resp)
```

## 8. WebSocket

- 市场频道（公开）：`wss://ws-subscriptions-clob.polymarket.com/ws/market`
  - 订阅：`{"assets_ids":[...], "type":"market"}`；动态增删 `operation: subscribe/unsubscribe`
  - **应用层心跳：每 10s 发文本帧 `PING`**
  - 事件：`book`（全量快照）、`price_change`（增量，size=0 删价位）、`last_trade_price`、`tick_size_change`（必须监听）
- 用户频道（私有）：`wss://ws-subscriptions-clob.polymarket.com/ws/user`
  - 订阅帧内嵌 `{"auth":{apiKey,secret,passphrase}, "markets":[condition_id], "type":"user"}`
  - 事件：`order`（PLACEMENT/UPDATE/CANCELLATION）、`trade`（MATCHED→MINED→CONFIRMED / RETRYING / FAILED）
  - **断线不回放**——重连后必须 REST 重拉 open orders + trades 重建状态
- Sports 实时：`wss://sports-api.polymarket.com/ws`（服务端 5s ping / 客户端 10s 内 pong）

## 9. 速率限制（双层）

- Cloudflare IP 级：超限节流排队（非拒绝）。`/book|/price|/midpoint` 各 150 req/s；`/prices-history` 100 req/s；`POST /order` 突发 500 / 持续 200。
- 按 signer 的 token bucket（2026-07-24 起强制）：Standard 档下单 40 tok/s（burst 60）、撤单 80（burst 120）；按 maker 30 天交易量升档（最高 Elite 600/900）。批量按笔数扣费且 all-or-nothing。429 时看 `Retry-After`；平时跟踪 `Poly-RateLimit-Remaining`。
- HTTP 425 = 撮合引擎重启（重启后约 2 分钟 post-only 模式），指数退避重试。

## 10. 交易前准备（一次性）

1. Polygon 链上 approve：pUSD → 两个 Exchange 合约；CTF `setApprovalForAll`（卖出前）。代理/Safe/Deposit 钱包自动处理，纯 EOA 需手动。
2. L1 派生 L2 凭据并持久化。
3. 卖出某 outcome token 前首次需 `balance-allowance/update`（CONDITIONAL）刷新缓存。

## 参考文档

- 全站索引：https://docs.polymarket.com/llms.txt
- 下单：https://docs.polymarket.com/trading/place-orders.md
- 钱包认证：https://docs.polymarket.com/trading/wallets-auth.md
- 费用：https://docs.polymarket.com/trading/fees.md
- Python SDK：https://docs.polymarket.com/getting-started/python.md
- SDK 迁移：https://docs.polymarket.com/getting-started/migrate-from-previous-sdks.md
- WS market/user：https://docs.polymarket.com/api-reference/wss/market.md 、/api-reference/wss/user.md
- 限速：https://docs.polymarket.com/api-reference/rate-limits.md 、/api-reference/trading-rate-limits.md
- OpenAPI：https://docs.polymarket.com/api-spec/clob-openapi.yaml 、gamma-openapi.yaml 、data-openapi.yaml
