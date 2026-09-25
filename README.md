# 低通滤波器设计与分析服务

常驻后端服务：FIR（窗函数法）/ IIR（巴特沃斯 + 双线性变换）系数设计、
单位圆频响求值、零极点分析，以及多段级联链路的合成与链路级指标。
HTTP 接口，JSON 进 JSON 出。

Python 3.12 + FastAPI，科学计算只借标准库（`math` / `cmath`）完成核心算法；
窗函数生成、双线性预畸变、相位展开、多项式求根、链路合成均为手写，
可单独测试，没有整段调用黑盒设计函数。

## 模块划分

| 文件 | 职责 |
| --- | --- |
| `app/windows.py` | 矩形 / 汉宁（hann、hanning 同义）/ 汉明窗，手写 |
| `app/fir.py` | FIR 窗函数法低通：理想冲激响应 × 窗 |
| `app/butterworth.py` | 巴特沃斯原型极点、频率预畸变、双线性变换，手写 |
| `app/polynomial.py` | 多项式乘法 / 求值，Aberth-Ehrlich 求根，手写 |
| `app/phase.py` | 主值相位与一维相位展开，手写 |
| `app/analysis.py` | 频响（dB + 展开相位）、零极点（到单位圆距离 + 稳定性） |
| `app/cascade/sections.py` | 段内求根、根配成共轭对、组织一阶 / 二阶实系数节 |
| `app/cascade/chain.py` | 段解析（系数段 / 设计意图段）、等效系数卷积、链式求值 |
| `app/cascade/metrics.py` | 链路级指标：纹波 / 阻带衰减 / 相位线性度与群延时 / 稳定性归因 |
| `app/cascade/service.py` | 链路请求编排与业务校验 |
| `app/cascade/api.py` | 级联链路 HTTP 路由（独立处理路径） |
| `app/validation.py` | 阶数 / 截止 / 系数 / 频点统一校验 |
| `app/schemas.py` | Pydantic 请求响应模型 |
| `app/main.py` | FastAPI 路由与统一错误处理 |

## 约定

- **阶数**：正整数，1..16（超过 16 拒绝）；抽头数 = 阶数 + 1。
- **截止频率**：相对奈奎斯特的归一化值，严格开区间 `(0, 1)`；
  对应数字角频率 `w = cutoff * pi`（rad/采样）。0 和 1（奈奎斯特）都拒绝。
- **系数格式**：FIR / IIR 统一返回 `{"b": [...], "a": [...]}`，
  按 z⁻¹ 升幂排列；IIR 的 `a[0]` 归一化为 1，FIR 的 `a = [1.0]`。
- **窗类型**：`rectangular`（`rect`/`boxcar`）、`hann`（`hanning` 同义）、
  `hamming`，别名大小写与首尾空格不敏感；不认识的名字返回 400。

## 接口

### `POST /api/v1/filters/design`

```json
// FIR
{"type": "fir", "order": 12, "cutoff": 0.3, "window": "hamming"}
// IIR
{"type": "iir", "order": 4, "cutoff": 0.25}
```

### `POST /api/v1/filters/frequency`

频率点为数字角频率（rad/采样），范围 `[0, pi]`；
返回每点 `magnitude_db` 与展开后的 `phase_rad`。

```json
{"b": [1.0], "a": [1.0, -0.7], "frequencies": [0.0, 0.5, 1.0]}
```

### `POST /api/v1/filters/zero-poles`

```json
{"b": [1.0], "a": [1.0, -0.7]}
```

返回零极点列表（实部 / 虚部 / 模 / 到单位圆距离 / 是否在单位圆上），
系统总增益 `gain`，以及整体 `stable`（存在单位圆外极点即为 `false`）。

### `POST /api/v1/cascade/analyze`

把一串滤波段当作整体合成与分析。`segments` 按链路顺序给出，每段二选一：

- `coefficients`：已设计好的 `{"b": [...], "a": [...]}`（格式同设计接口输出）；
- `design`：设计意图 `{"type", "order", "cutoff", "window?"}`，服务现场设计后入链。

`passband` / `stopband` 为 `[下限, 上限]`（相对奈奎斯特，`[0, 1]` 闭区间），
两者同时给出才会计算链路级指标；`grid_points` 为指标采样点数
（整数 32..8192，缺省 4096）。

```json
{
  "segments": [
    {"name": "压工频", "design": {"type": "iir", "order": 4, "cutoff": 0.2}},
    {"name": "削毛刺", "design": {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"}},
    {"name": "整形", "coefficients": {"b": [1.0, -0.5], "a": [1.0]}}
  ],
  "passband": [0.0, 0.1],
  "stopband": [0.45, 1.0]
}
```

返回：

- `equivalent`：等效分子分母系数 `b` / `a`（各段传递函数相乘，即分子、
  分母分别卷积）与阶数；格式与单滤波器完全一致，可直接喂回
  `/api/v1/filters/frequency`、`/api/v1/filters/zero-poles` 复核；
- `segments`：每段的序号、名字、来源（design / coefficients）、
  规范化设计参数与落定系数，按链路次序排列，方便对账；
- `metrics`（给了频带时）：通带纹波（相对直流参考电平的最大偏差与
  峰峰值，dB）、阻带衰减（参考电平减去阻带最高增益，dB）、相位
  （通带展开相位对最佳线性拟合的最大 / 均方根残差、是否接近线性、
  等效群延时采样数、是否含 IIR 段）、稳定性（整体是否稳定、
  由哪几段引入单位圆外极点、逐段逐极点归因列表）。

数值约定：合成后的等效多项式可达几十阶，内部**不**对它直接求根或
代入单位圆求值（高阶展开式会把圆内极点算到圆外、深阻带相消出假值）。
指标与极点一律逐段计算再合成：求值走"逐段 Horner + 复数相乘"，
求根走"段内求根 + 共轭对配节"，与等效传递函数数学等价但数值稳定。

所有业务错误（非法阶数 / 截止、空频点数组、频点或系数里掺非数值等）
统一返回 HTTP 400，形如 `{"error": "中文原因"}`。

## 运行

```bash
docker build -t filter-service .
docker run --rm -p 8000:8000 filter-service
# 或本地
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

自动化测试盯住以下关系：

- FIR 系数首尾对称（浮点容差内），奇数 / 偶数阶切换时抽头数与对称中心跟着变；
- 预畸变与反预畸变互相反推；并构造一份"漏做预畸变"的对照实现，
  其真实 -3dB 交叉点明显偏离目标，防止用未预畸变的值糊弄；
- FIR 通带内展开相位斜率 = -order/2（群延时）；
- 同一组系数，频响直接代入与"零极点求根后重组"两条路径在截止附近
  幅度 dB 与衰减趋势一致；
- IIR 极点全部在单位圆内、`a[0]=1`、指定截止处约 -3.01 dB；
- 单位圆外极点明确标 `stable=false`，圆上极点单独标注；
- 阶数 0 / 负数 / >16、截止 0 / ≥1、未知窗名、空频点、非数值元素等一律拒绝。

级联链路（`tests/test_cascade.py` / `tests/test_cascade_internals.py`）：

- 等效系数正确性：等效 b/a 进频响接口，与"每段单独求频响再逐点
  复数相乘"在浮点容差内一致；
- 级联顺序无关性：同样几段换次序，等效系数（容差内）与等效频响、
  链路指标完全一致；
- 数值稳定性：四段 6 阶紧截止 IIR 串成 24 阶链路，全部极点仍判在
  单位圆内，并与解析极点逐根对上；反面事实也被固定——对合成后的
  24 阶分母直接求根会把圆内极点推到圆外（这正是逐段求根的原因）；
- 退化情形：单段链路的等效系数与该段逐位相等（不是近似）；
- 阻带衰减方向性：再串一段同型低通，指定阻带的最坏衰减只深不浅；
- 非法输入一律 400 + 中文原因：空链路、系数掺非数值、分母全零、
  设计意图阶数 / 截止越界、频带给反或越界、采样点数非法等；
- 指标行为：纯 FIR 链相位线性且群延时等于各段之和，混入 IIR 后
  如实报告非线性偏离，不稳定段按序号点名，节分解能还原原段系数。
