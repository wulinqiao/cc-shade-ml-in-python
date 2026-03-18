# CC-SHADE-ML

**自适应协同进化大规模全局优化算法复现**
**Reproduction of Self-Adaptive Cooperative Coevolution for Large-Scale Global Optimization**

---

## 算法简介 / Overview

CC-SHADE-ML 是一种用于求解大规模全局优化问题（LSGO）的元启发式算法。
CC-SHADE-ML is a metaheuristic algorithm for solving Large-Scale Global Optimization (LSGO) problems.

算法将三种机制结合 / The algorithm combines three mechanisms:

- **CC（协同进化 / Cooperative Coevolution）**：将 1000 维问题随机分解为若干子分量，轮流优化。Decomposes the 1000-dimensional problem into subcomponents optimized in rotation.
- **SHADE（自适应差分进化 / Success-History based Adaptive DE）**：每个子分量的优化器，自适应调整缩放因子 F 和交叉率 CR。Subcomponent optimizer with adaptive F and CR via history memory.
- **ML（多层次自适应 / Multi-Level adaptation）**：在运行过程中根据历史性能动态选择子分量数量和种群大小。Dynamically selects subcomponent count and population size based on historical performance.

> 参考论文 / Reference: Vakhnin & Sopov, *A Novel Self-Adaptive Cooperative Coevolution Algorithm for Solving Continuous Large-Scale Global Optimization Problems*, Algorithms 2022, 15, 451.

---

## 文件结构 / File Structure

```
cc-shade-ml  Philip/
├── benchmark/
│   └── cec2013lsgo/
│       ├── datafiles/      # CEC'2013 数据文件 / CEC'2013 data files
│       ├── cec2013.py      # Benchmark 入口 / Benchmark entry
│       ├── benchmarks.py
│       └── f1.py ~ f15.py
└── cc-shade-ml/            # 本算法 / This algorithm
    ├── constants.py        # 随机数生成 / Random number generation
    ├── header.py           # 算法核心工具函数 / Core algorithm utilities
    ├── run_ccshademl.py    # 主运行入口 / Main runner
    ├── test.py             # 单元测试 / Unit tests
    └── README.md
```

`run_ccshademl.py` 的内部结构 / Internal structure of `run_ccshademl.py`:

| 组件 / Component | 说明 / Description |
|---|---|
| `_TrialState` | 单次实验状态容器，封装全部工作数组 / State container encapsulating all working arrays |
| `call_fun()` | 目标函数调用适配器 / Objective function call adapter |
| `_allocate_arrays()` | 分配全部工作数组 / Allocate all working arrays |
| `_setup_outer_cycle()` | 每轮外循环的变量分组和重置 / Variable grouping and reset per outer cycle |
| `_init_subcomp_fitness()` | 初始化子分量适应度（构建上下文向量）/ Initialize subcomponent fitness |
| `_mutate_crossover()` | 单个体的变异和交叉 / Mutation and crossover for one individual |
| `_eval_and_select()` | 评估试验向量并贪心选择 / Evaluate trial vector and greedy selection |
| `_run_subcomp_shade()` | 单子分量完整 SHADE 步骤 / Full SHADE step on one subcomponent |
| `_run_outer_cycle()` | 组织一次完整外循环 / Orchestrate one complete outer cycle |
| `_update_performance()` | 更新 ML 性能分数 / Update ML performance scores |
| `run_one_trial()` | 公开接口，驱动完整实验 / Public interface, drives full trial |
| `main()` | 命令行入口 / CLI entry point |

---

## 环境依赖 / Requirements

```
Python >= 3.11
numpy
numba
```

---

## 快速开始 / Quick Start

所有命令在根目录（`cc-shade-ml  Philip/`）下运行 / Run all commands from the root directory (`cc-shade-ml  Philip/`).

**测试单个函数 / Test a single function:**

```bash
python cc-shade-ml/run_ccshademl.py --id 1 --runs 1 --fev 50000
```

**完整实验（单个函数，25次运行）/ Full experiment (single function, 25 runs):**

```bash
python cc-shade-ml/run_ccshademl.py --id 1 --runs 25 --fev 3000000
```

**运行所有 15 个函数 / Run all 15 functions:**

```bash
python cc-shade-ml/run_ccshademl.py --runs 25 --fev 3000000
```

**运行单元测试 / Run unit tests:**

```bash
python cc-shade-ml/test.py
```

---

## 命令行参数 / Command Line Arguments

| 参数 / Argument | 类型 / Type | 默认值 / Default | 说明 / Description |
|---|---|---|---|
| `--id` | int | None（全跑）| 函数编号 1-15 / Function ID 1-15 |
| `--runs` | int | 25 | 独立运行次数 / Number of independent runs |
| `--fev` | int | 3,000,000 | 函数评估预算 / Function evaluation budget |

---

## 算法参数 / Algorithm Parameters

以下参数与原论文 tuned 版本保持一致 / Parameters consistent with the tuned version in the paper:

| 参数 / Parameter | 值 / Value | 说明 / Description |
|---|---|---|
| 子分量候选池 / Subcomponent pool | {5, 10, 20, 50} | 每轮从中自适应选择 / Adaptively selected each round |
| 种群大小候选池 / Population size pool | {25, 50, 100} | 每轮从中自适应选择 / Adaptively selected each round |
| Boltzmann 温度 / Boltzmann temperature | 7.0 | 控制选择压力 / Controls selection pressure |
| SHADE 历史记忆长度 / SHADE history length | 6 | 存储成功参数的条数 / Number of successful parameters stored |
| 档案大小 / Archive size | 100 | 外部档案容量 / External archive capacity |
| 内循环预算 / Inner loop budget | FEV_total / 50 | 每次自适应选择后的优化步数 / Steps per adaptive selection |

---

## 输出说明 / Output

结果保存在 `results/` 文件夹下 / Results saved in the `results/` folder:

```
results/
└── f1_ccshademl.txt    # 每个函数一个文件 / One file per function
```

每个文件包含 / Each file contains:
- 每次运行的 100 个等间隔 best-so-far 记录点 / 100 evenly-spaced best-so-far checkpoints per run
- 25 次运行的统计汇总：BEST、MEAN、MEDIAN、STD、WORST / Statistical summary over 25 runs

对应论文记录点 / Checkpoints matching the paper: `1.2×10⁵`、`6.0×10⁵`、`3.0×10⁶`

---

## 接口说明 / Interface

与 demo 框架的接口保持一致 / Interface consistent with the demo framework:

```python
from benchmark.cec2013lsgo.cec2013 import Benchmark
from cc_shade_ml.run_ccshademl import run_one_trial

benchmark = Benchmark()
fun  = benchmark.get_function(1)   # 获取目标函数 / Get objective function
info = benchmark.get_info(1)       # 获取函数信息 / Get function info
# info = {'lower': -100, 'upper': 100, 'dimension': 1000}

record, elapsed = run_one_trial(fun, info, FEV_global=3000000, seed=42)
# record: 100个 best-so-far 值 / 100 best-so-far values
# elapsed: 运行时间（秒）/ Wall-clock time in seconds
```

---

## 测试说明 / Test Description

`test.py` 使用 `unittest` 框架，覆盖以下组件 / `test.py` uses the `unittest` framework, covering:

| 测试类 / Test Class | 覆盖内容 / Coverage |
|---|---|
| `TestRandomGeneration` | 随机数范围、均匀性、种子可复现性 / Range, uniformity, seed reproducibility |
| `TestPopulationInit` | 种群形状、边界、多样性 / Shape, bounds, diversity |
| `TestVariableGrouping` | 置换完整性、随机性 / Permutation completeness, randomness |
| `TestSHADEParameters` | F/CR 采样范围、历史初始化 / F/CR sampling range, history initialization |
| `TestBorderCheck` | 越界修复正确性 / Border repair correctness |
| `TestBoltzmannSelection` | 高性能配置优先选择、均匀性 / High-performance preference, uniformity |
| `TestHistoryUpdate` | 成功/无成功时的历史更新行为 / History update on success/no-success |
| `TestBenchmarkInterface` | 函数接口字段、维度、返回值 / Interface fields, dimension, return value |
| `TestSmokeRun` | 端到端可运行、收敛单调性 / End-to-end execution, monotone convergence |

预期输出 / Expected output:
```
Ran 30 tests in ~40s
OK
```
