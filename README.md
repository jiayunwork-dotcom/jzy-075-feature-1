# 低通滤波器设计与分析服务

常驻后端服务：FIR（窗函数法）/ IIR（巴特沃斯 + 双线性变换）系数设计、
单位圆频响求值、零极点分析，以及**多段级联链路的等效合成与链路级指标**。
四个 HTTP 接口，JSON 进 JSON 出。

Python 3.12 + FastAPI，科学计算只借标准库（`math` / `cmath`）完成核心算法；
窗函数生成、双线性预畸变、相位展开、多项式求根均为手写，可单独测试，
没有整段调用黑盒设计函数。

## 模块划分

| 文件 | 职责 |
| --- | --- |
| `app/windows.py` | 矩形 / 汉宁（hann、hanning 同义）/ 汉明窗，手写 |
| `app/fir.py` | FIR 窗函数法低通：理想冲激响应 × 窗 |
| `app/butterworth.py` | 巴特沃斯原型极点、频率预畸变、双线性变换，手写 |
| `app/polynomial.py` | 多项式乘法 / 求值，Aberth-Ehrlich 求根，手写 |
| `app/phase.py` | 主值相位与一维相位展开，手写 |
| `app/analysis.py` | 频响（dB + 展开相位）、零极点（到单位圆距离 + 稳定性） |
| `app/cascade.py` | 级联链路段解析、等效卷积合成、逐段稳健求值与极点归属 |
| `app/chain_metrics.py` | 链路级指标：通带纹波 / 阻带衰减 / 相位线性度与群延时 |
| `app/validation.py` | 阶数 / 截止 / 系数 / 频点统一校验 |
| `app/schemas.py` | Pydantic 请求响应模型 |
| `app/main.py` | FastAPI 路由与统一错误处理（单滤波器） |
| `app/cascade_api.py` | 级联链路独立路由 `POST /api/v1/filters/cascade` |

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

### `POST /api/v1/filters/cascade`

级联链路独立路径。`segments` 按顺序给出，每段要么是已落地系数
（`type: "coefficients"` + `b/a`），要么是现场设计的设计意图
（`type: "fir"` 带 `window`，或 `type: "iir"` 巴特沃斯）；
`passband` / `stopband` 为相对奈奎斯特的归一化区间 `[下限, 上限]`。

```json
{
  "segments": [
    {"type": "iir", "order": 4, "cutoff": 0.2},
    {"type": "fir", "order": 12, "cutoff": 0.35, "window": "hamming"},
    {"type": "coefficients", "b": [1.0, -0.5], "a": [1.0]}
  ],
  "passband": [0.0, 0.15],
  "stopband": [0.45, 1.0]
}
```

返回：

- `b` / `a`：各段分子、分母分别卷积后的**等效单一系统系数**，
  格式与 `/design` 完全一致，可直接喂回 `/frequency`、`/zero-poles` 复核；
- `segments`：各段落地后的系数、类型、次序与设计意图（`spec`），用于对账；
- `metrics`（全部基于合成后的整条链路，不做各段指标简单相加）：
  - `passband_ripple_db`：通带增益相对带内参考电平（最大增益）的最大起伏（dB）；
  - `stopband_attenuation_db`：指定阻带内最坏（最不衰减）点相对参考电平的压制 dB；
  - `linear_phase`、`phase_max_deviation_rad`、`phase_rms_deviation_rad`：
    通带内展开相位对最佳线性拟合的偏离；混进 IIR 后如实报 `false`，不假装线性；
  - `group_delay_samples`、`group_delay_variation_samples`：
    通带平均群延时（采样数）及其起伏；
  - `stable`、`unstable_segments`、`unstable_poles`：逐段分母求根判定，
    圆外极点明确归属到引入它的段号。

**数值守护**：链路指标不在二三十阶展开多项式上直接求值 / 求根
（高次项与低次项量级悬殊，实测会让通带读数崩到 -200 dB 以下、
把单位圆内极点误算到圆外）。等效系数仍按卷积如实给出供对账；
指标计算改为每段（阶数 ≤ 16）各自 Horner 代入、dB 域逐段累加
（深阻带连乘会下溢）、相位逐段相加后统一展开，极点逐段求根，
与"每段单独算再逐点相乘"的参照结果在深阻带 -300 dB 量级仍一致。

所有业务错误（空链路、非数值系数、全零分母、阶数 / 截止越界、
频带给反或超出 `[0, 1]` 等）统一返回 HTTP 400，形如
`{"error": "中文原因"}`；请求体结构性错误（缺字段 / 类型不对）为 422。

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

级联链路（`tests/test_cascade.py`、`tests/test_cascade_api.py`）逐条盯住：

1. **等效系数正确性**：合成的等效 b/a 拿回频响接口求值，与"每段单独求频响、
   复数响应逐点相乘"一致（浮点容差随衰减深度缩放）；
2. **顺序无关性**：同样几段换次序，等效频响 / 系数在浮点容差内完全一致；
3. **数值稳定性**：三段 IIR 串到 38 阶，逐段求根所有极点仍判在单位圆内；
   同时钉死对照事实——同一展开分母直接求根会把十余个稳定极点误判到圆外
   （半径 > 1.3）、展开 Horner 会把通带读数算成 -200 dB 以下，证明分段不是多余；
4. **退化情形**：只有一段时等效 b/a 与该段逐位相等；
5. **阻带衰减方向性**：再串一段低通，指定阻带最坏衰减只深不浅；
   两段相同 IIR 时衰减按 dB 叠加（约两倍）；
6. **非法输入**：空链路、系数掺非数值 / 分母全零、阶数截止越界、
   通带阻带给反或超出 `[0, 1]` 一律 400 并给中文原因。
   另有 IIR 混入后 `linear_phase=false` 并给偏离量、纯 FIR 链群延时
   等于各段 order/2 之和、圆外极点归属到具体段号等指标级断言。
