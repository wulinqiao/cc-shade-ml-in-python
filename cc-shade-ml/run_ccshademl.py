"""
run_ccshademl.py — CC-SHADE-ML 算法主运行模块 / CC-SHADE-ML Algorithm Main Runner.

此模块实现 CC-SHADE-ML 算法与 demo 基准测试框架的集成。
This module integrates the CC-SHADE-ML algorithm with the demo benchmark framework.

主要组件 / Main Components:
    - _TrialState:             单次实验状态容器 / State container for one trial
    - call_fun():              目标函数调用适配器 / Objective function adapter
    - run_one_trial():         单次独立实验 / Single independent trial
    - main():                  命令行入口 / CLI entry point

使用示例 / Usage Example:
    python run_ccshademl.py --id 1 --runs 25 --fev 3000000

参考 / References:
    - Vakhnin & Sopov, CC-SHADE-ML, Algorithms 2022, 15, 451
"""

import os
import sys
import time
import math
import logging
import argparse
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, List, Optional, Tuple

# ── 路径修复 / Path Fix ───────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from benchmark.cec2013lsgo.cec2013 import Benchmark
import constants as C
from header import (
    initializePopulation, initializeHistory,
    findBestIndex, generation_CR, generation_F,
    chooseCrossoverIndecies, Algorithm_1,
    check_out_borders, updateArchive,
    find_best_part_index, find_best_fitness_value,
    rnd_indecies, indecesSuccession, randperm,
    random_performance, max_number,
    mean_stat, min_stat, max_stat, median_stat, stddev_stat,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 算法参数常量 / Algorithm Parameter Constants
# ══════════════════════════════════════════════════════════════════════════════

DIMENSION       = 1000          # 问题维度 / Problem dimension
FEV_GLOBAL      = int(3e6)      # 函数评估预算 / Function evaluation budget
R_RUNS          = 25            # 独立运行次数 / Number of independent runs
HISTORY_LENGTH  = 6             # SHADE 历史记忆长度 / SHADE history length
PIECE           = 0.1           # pbest 比例 / Fraction for pbest selection
POWER           = 7.0           # Boltzmann 温度（论文推荐值）/ Boltzmann temperature

# 多层次候选池（tuned 版本）/ Multi-level candidate pools (tuned version)
POP_SIZE_POOL       = [25, 50, 100]
SUBCOMPONENTS_POOL  = [5, 10, 20, 50]

POP_SIZE_MAX  = max(POP_SIZE_POOL)      # 100
ARCHIVE_SIZE  = POP_SIZE_MAX            # 100
M_MAX         = max(SUBCOMPONENTS_POOL) # 50


# ══════════════════════════════════════════════════════════════════════════════
# 状态容器 / State Container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TrialState:
    """
    单次实验的完整可变状态 / Complete mutable state for one trial run.

    封装所有工作数组和运行时标量，使各辅助函数仅需 state 作为主参数。
    Encapsulates all working arrays and runtime scalars so helper functions
    only need state as their primary parameter.
    """
    population:               npt.NDArray[np.float64]
    population_new:           npt.NDArray[np.float64]
    archive:                  npt.NDArray[np.float64]
    solution:                 npt.NDArray[np.float64]
    u:                        npt.NDArray[np.float64]
    history_f:                List[List[float]]
    history_cr:               List[List[float]]
    f_arr:                    List[float]
    cr_arr:                   List[float]
    s_cr:                     List[float]
    s_f:                      List[float]
    delta_f:                  List[float]
    w_arr:                    List[float]
    cc_best_individual_index: List[int]
    fitness_cc:               List[List[float]]
    fitness_cc_new:           List[List[float]]
    range_arr:                List[int]
    k:                        List[int]
    A:                        List[int]
    indeces:                  List[int]
    performance_cc:           List[float]
    performance_pop_size:     List[float]
    FEV:                      int
    fev_cycle:                int
    best_solution:            float
    fitness_record:           List[float]
    trigger:                  int
    FEV_global:               int
    krantost:                 int
    a_bound:                  float
    b_bound:                  float
    M:                        int = 0
    pop_size:                 int = 0
    piece_int:                int = 0


# ══════════════════════════════════════════════════════════════════════════════
# 目标函数适配器 / Objective Function Adapter
# ══════════════════════════════════════════════════════════════════════════════

def call_fun(
    fun: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64]
) -> float:
    """
    目标函数调用适配器 / Objective function call adapter.

    demo 框架返回 shape(1,) 数组，本函数统一转为 Python float，
    并传副本防止 numba JIT 原地修改输入。
    The demo framework returns shape(1,) arrays; this converts to float
    and passes a copy to prevent in-place modification by numba JIT.

    Args:
        fun (Callable): demo 框架目标函数 / Demo framework objective function
        x (ndarray):    解向量 shape(N,) / Solution vector shape(N,)

    Returns:
        float: 目标函数值 / Objective function value
    """
    result = fun(x.copy())
    if hasattr(result, '__len__'):
        return float(result[0])
    return float(result)


# ══════════════════════════════════════════════════════════════════════════════
# 内部辅助函数 / Internal Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _allocate_arrays(FEV_global: int, a_bound: float, b_bound: float) -> _TrialState:
    """
    分配并初始化一次实验所需的全部工作数组 / Allocate all working arrays for one trial.

    Args:
        FEV_global (int): 函数评估总预算 / Total function evaluation budget
        a_bound (float):  搜索空间下界 / Search space lower bound
        b_bound (float):  搜索空间upper bound

    Returns:
        _TrialState: 初始化后的状态对象 / Initialized state object
    """
    return _TrialState(
        population               = np.zeros((POP_SIZE_MAX, DIMENSION)),
        population_new           = np.zeros((POP_SIZE_MAX, DIMENSION)),
        archive                  = np.zeros((ARCHIVE_SIZE, DIMENSION)),
        solution                 = np.zeros(DIMENSION),
        u                        = np.zeros((POP_SIZE_MAX, DIMENSION)),
        history_f                = [[0.0] * HISTORY_LENGTH for _ in range(M_MAX)],
        history_cr               = [[0.0] * HISTORY_LENGTH for _ in range(M_MAX)],
        f_arr                    = [0.0] * POP_SIZE_MAX,
        cr_arr                   = [0.0] * POP_SIZE_MAX,
        s_cr                     = [0.0] * POP_SIZE_MAX,
        s_f                      = [0.0] * POP_SIZE_MAX,
        delta_f                  = [0.0] * POP_SIZE_MAX,
        w_arr                    = [0.0] * POP_SIZE_MAX,
        cc_best_individual_index = [0] * M_MAX,
        fitness_cc               = [[0.0] * POP_SIZE_MAX for _ in range(M_MAX)],
        fitness_cc_new           = [[0.0] * POP_SIZE_MAX for _ in range(M_MAX)],
        range_arr                = [0] * (M_MAX + 1),
        k                        = [0] * M_MAX,
        A                        = [0] * M_MAX,
        indeces                  = list(range(DIMENSION)),
        performance_cc           = [1.0] * len(SUBCOMPONENTS_POOL),
        performance_pop_size     = [1.0] * len(POP_SIZE_POOL),
        FEV                      = FEV_global,
        fev_cycle                = 0,
        best_solution            = 1e300,
        fitness_record           = [],
        trigger                  = 0,
        FEV_global               = FEV_global,
        krantost                 = FEV_global // 100,
        a_bound                  = a_bound,
        b_bound                  = b_bound,
    )


def _setup_outer_cycle(state: _TrialState) -> None:
    """
    设置每轮外循环的变量分组和计数器 / Setup variable grouping and counters for each outer cycle.

    执行随机置换、建立子分量边界、重置档案计数和历史记忆。
    Performs random permutation, builds subcomponent boundaries, resets
    archive counts and history memory.

    Args:
        state (_TrialState): 当前实验状态 / Current trial state
    """
    S = DIMENSION // state.M
    state.range_arr[0] = 0
    state.range_arr[state.M] = DIMENSION
    for i in range(1, state.M):
        state.range_arr[i] = state.range_arr[i - 1] + S
    indecesSuccession(state.indeces, DIMENSION)
    randperm(state.indeces, DIMENSION)
    for i in range(state.M):
        state.A[i] = 0
        state.k[i] = 0
    for i in range(state.pop_size):
        state.s_cr[i] = state.s_f[i] = state.delta_f[i] = 0.0
    initializeHistory(state.history_f, state.history_cr, HISTORY_LENGTH, state.M)
    rnd_indecies(state.cc_best_individual_index, state.pop_size, state.M)


def _init_subcomp_fitness(state: _TrialState, fun: Callable) -> None:
    """
    初始化各子分量的适应度值 / Initialize fitness values for all subcomponents.

    通过构造上下文向量评估每个个体在每个子分量中的适应度。
    Evaluates each individual in each subcomponent via context vector construction.

    Args:
        state (_TrialState): 当前实验状态 / Current trial state
        fun (Callable):      目标函数 / Objective function
    """
    for p in range(state.M):
        for i in range(state.pop_size):
            for j in range(state.range_arr[p], state.range_arr[p + 1]):
                state.solution[state.indeces[j]] = state.population[i][state.indeces[j]]
            for p_cc in range(state.M):
                if p_cc != p:
                    bi = state.cc_best_individual_index[p_cc]
                    for j in range(state.range_arr[p_cc], state.range_arr[p_cc + 1]):
                        state.solution[state.indeces[j]] = state.population[bi][state.indeces[j]]
            val = call_fun(fun, state.solution)
            state.fitness_cc[p][i] = val
            state.fitness_cc_new[p][i] = val
            state.FEV -= 1
            if state.FEV % state.krantost == 0 and state.trigger < 100:
                state.fitness_record.append(state.best_solution)
                state.trigger += 1
            if val < state.best_solution:
                state.best_solution = val
        find_best_part_index(state.cc_best_individual_index, state.fitness_cc, p, state.pop_size)
    best_v = find_best_fitness_value(state.fitness_cc, state.M, state.pop_size)
    if best_v < state.best_solution:
        state.best_solution = best_v


def _mutate_crossover(state: _TrialState, p: int, i: int) -> None:
    """
    对个体 i 在子分量 p 上执行变异和交叉 / Mutation and crossover for individual i in subcomponent p.

    使用 current-to-pbest/1 变异策略和二项式交叉，之后执行越界修复。
    Uses current-to-pbest/1 mutation and binomial crossover, followed by border repair.

    Args:
        state (_TrialState): 当前实验状态 / Current trial state
        p (int): 子分量索引 / Subcomponent index
        i (int): 个体索引 / Individual index
    """
    pbest = findBestIndex(state.fitness_cc, state.pop_size, state.piece_int, p)
    r_idx = int(C.RANDOM() * (HISTORY_LENGTH - 1))
    state.cr_arr[i] = generation_CR(state.history_cr, r_idx, p)
    state.f_arr[i]  = generation_F(state.history_f,  r_idx, p)
    r1, r2 = chooseCrossoverIndecies(pbest, state.pop_size, state.A, p)
    src   = state.population if r2 < state.pop_size else state.archive
    r2idx = r2 if r2 < state.pop_size else r2 - state.pop_size
    fi = state.f_arr[i]
    for j in range(state.range_arr[p], state.range_arr[p + 1]):
        idx = state.indeces[j]
        state.u[i][idx] = (state.population[i][idx]
            + fi * (state.population[pbest][idx] - state.population[i][idx])
            + fi * (state.population[r1][idx]    - src[r2idx][idx]))
    jrand = int(C.RANDOM() * ((state.range_arr[p+1] - state.range_arr[p]) + state.range_arr[p]))
    for j in range(state.range_arr[p], state.range_arr[p + 1]):
        idx = state.indeces[j]
        if not (C.RANDOM() <= state.cr_arr[i] or j == jrand):
            state.u[i][idx] = state.population[i][idx]
    check_out_borders(state.u, state.population, i, state.a_bound, state.b_bound,
                      state.range_arr, p, state.indeces)


def _eval_and_select(state: _TrialState, p: int, i: int, fun: Callable, success: int) -> int:
    """
    评估试验向量并执行贪心选择 / Evaluate trial vector and perform greedy selection.

    构造上下文向量后评估，若优于父代则更新种群并记录成功参数。
    Constructs context vector, evaluates, updates population if improved.

    Args:
        state (_TrialState): 当前实验状态 / Current trial state
        p (int):     子分量索引 / Subcomponent index
        i (int):     个体索引 / Individual index
        fun:         目标函数 / Objective function
        success (int): 当前成功计数 / Current success count

    Returns:
        int: 更新后的成功计数 / Updated success count
    """
    for j in range(state.range_arr[p], state.range_arr[p + 1]):
        state.solution[state.indeces[j]] = state.u[i][state.indeces[j]]
    for p_cc in range(state.M):
        if p_cc != p:
            bi = state.cc_best_individual_index[p_cc]
            for j in range(state.range_arr[p_cc], state.range_arr[p_cc + 1]):
                state.solution[state.indeces[j]] = state.population[bi][state.indeces[j]]
    test_val = call_fun(fun, state.solution)
    state.FEV      -= 1
    state.fev_cycle -= 1
    if state.FEV % state.krantost == 0 and state.trigger < 100:
        state.fitness_record.append(state.best_solution)
        state.trigger += 1
    if test_val < state.fitness_cc[p][i]:
        for j in range(state.range_arr[p], state.range_arr[p + 1]):
            state.population_new[i][state.indeces[j]] = state.u[i][state.indeces[j]]
        updateArchive(state.archive, state.population, i, ARCHIVE_SIZE,
                      state.A, state.range_arr, p, state.indeces)
        state.fitness_cc_new[p][i] = test_val
        state.delta_f[success]     = abs(test_val - state.fitness_cc[p][i])
        state.s_f[success]         = state.f_arr[i]
        state.s_cr[success]        = state.cr_arr[i]
        return success + 1
    return success


def _run_subcomp_shade(state: _TrialState, p: int, fun: Callable) -> None:
    """
    对子分量 p 执行完整的 SHADE 优化步骤 / Run one full SHADE step on subcomponent p.

    依次执行变异交叉、评估选择、历史记忆更新和种群更新。
    Sequentially: mutation+crossover, evaluation+selection,
    history memory update, population update.

    Args:
        state (_TrialState): 当前实验状态 / Current trial state
        p (int):    子分量索引 / Subcomponent index
        fun:        目标函数 / Objective function
    """
    success = 0
    for i in range(state.pop_size):
        _mutate_crossover(state, p, i)
    for i in range(state.pop_size):
        success = _eval_and_select(state, p, i, fun, success)
    Algorithm_1(state.delta_f, state.w_arr, state.s_cr, state.s_f,
                state.history_cr, state.history_f, state.k, success, HISTORY_LENGTH, p)
    for i in range(state.pop_size):
        for j in range(state.range_arr[p], state.range_arr[p + 1]):
            state.population[i][state.indeces[j]] = state.population_new[i][state.indeces[j]]
        state.fitness_cc[p][i] = state.fitness_cc_new[p][i]
    min_pop_f = find_best_fitness_value(state.fitness_cc, state.M, state.pop_size)
    if min_pop_f < state.best_solution:
        state.best_solution = min_pop_f
    find_best_part_index(state.cc_best_individual_index, state.fitness_cc, p, state.pop_size)


def _update_performance(
    state: _TrialState,
    best_before: float,
    cc_index: int,
    pop_size_index: int
) -> None:
    """
    更新 ML 层的性能分数（论文公式2）/ Update ML performance scores (paper Eq. 2).

    计算本轮相对改进量，防止极端值后写回 performance 向量。
    Computes relative improvement, guards against extreme values,
    then writes back to performance vectors.

    Args:
        state (_TrialState):  当前实验状态 / Current trial state
        best_before (float):  内循环前的最优值 / Best value before inner loop
        cc_index (int):       本轮选中的子分量配置索引 / Selected subcomponent config index
        pop_size_index (int): 本轮选中的种群大小配置索引 / Selected population size config index
    """
    best_after = state.best_solution
    perf = (best_before - best_after) / abs(best_after) if best_after != 0.0 else 0.0
    if math.isinf(perf) or math.isnan(perf):
        perf = 1e4
    perf = min(perf, 100.0)
    if perf < 1e-4:
        perf = 1e-4
    state.performance_cc[cc_index]             = perf
    state.performance_pop_size[pop_size_index] = perf


def _print_and_save_results(
    fun_id: int,
    finals: List[float],
    all_records: List[List[float]],
    all_times: List[float],
    args: argparse.Namespace
) -> None:
    """
    打印统计汇总并保存结果到文件 / Print statistical summary and save results to file.

    Args:
        fun_id (int):    函数编号 / Function ID
        finals (List):   各次运行最终最优值 / Final best value per run
        all_records:     各次运行收敛记录 / Convergence records per run
        all_times:       各次运行耗时 / Elapsed time per run
        args:            命令行参数 / CLI arguments
    """
    import numpy as np
    print(f"\n  [f{fun_id} 汇总 / Summary]")
    print(f"  BEST  : {min(finals):.6e}")
    print(f"  MEAN  : {np.mean(finals):.6e}")
    print(f"  MEDIAN: {np.median(finals):.6e}")
    if args.runs > 1:
        print(f"  STD   : {np.std(finals, ddof=1):.6e}")
    print(f"  WORST : {max(finals):.6e}")
    print(f"  平均时间 / Avg time: {np.mean(all_times):.1f}s / run")
    out_path = f"results/f{fun_id}_ccshademl.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"CC-SHADE-ML  f{fun_id}  runs={args.runs}  FEV={args.fev}\n\n")
        f.write("100 best-so-far checkpoints per run / 每次运行100个记录点:\n")
        for z, rec in enumerate(all_records):
            f.write(f"Run {z+1:2d}: " + ", ".join(f"{v:.4e}" for v in rec) + "\n")
        f.write(f"\nBEST  : {min(finals):.6e}\n")
        f.write(f"MEAN  : {np.mean(finals):.6e}\n")
        f.write(f"MEDIAN: {np.median(finals):.6e}\n")
        if args.runs > 1:
            f.write(f"STD   : {np.std(finals, ddof=1):.6e}\n")
        f.write(f"WORST : {max(finals):.6e}\n")
    print(f"  结果已保存 / Results saved: {out_path}")
    logger.info(f"f{fun_id} done: mean={np.mean(finals):.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# 公开接口 / Public Interface
# ══════════════════════════════════════════════════════════════════════════════

def _run_outer_cycle(state: _TrialState, fun: Callable) -> None:
    """
    执行一次完整的外循环周期 / Execute one complete outer cycle.

    选择配置→分组→初始化适应度→内循环 SHADE→更新性能分数。
    Selects config, groups variables, initialises fitness, runs inner SHADE
    loop, then updates performance scores.

    Args:
        state (_TrialState): 当前实验状态 / Current trial state
        fun (Callable):      目标函数 / Objective function
    """
    cc_idx       = random_performance(state.performance_cc, len(SUBCOMPONENTS_POOL), POWER)
    pop_size_idx = random_performance(state.performance_pop_size, len(POP_SIZE_POOL), POWER)
    state.M         = SUBCOMPONENTS_POOL[cc_idx]
    state.pop_size  = POP_SIZE_POOL[pop_size_idx]
    state.piece_int = max(1, int(state.pop_size * PIECE))
    _setup_outer_cycle(state)
    _init_subcomp_fitness(state, fun)
    best_before     = state.best_solution
    state.fev_cycle = state.FEV_global // 50
    while state.fev_cycle > 0 and state.FEV > 0:
        for p in range(state.M):
            _run_subcomp_shade(state, p, fun)
    _update_performance(state, best_before, cc_idx, pop_size_idx)


def run_one_trial(
    fun: Callable[[npt.NDArray[np.float64]], float],
    info: Dict,
    FEV_global: int,
    seed: Optional[int] = None
) -> Tuple[List[float], float]:
    """
    运行一次 CC-SHADE-ML 独立实验 / Run one independent CC-SHADE-ML trial.

    实现论文 Algorithm 2：反复调用外循环直到预算耗尽。
    Implements paper Algorithm 2: repeatedly calls outer cycle until budget exhausted.

    Args:
        fun (Callable):       demo 框架目标函数 / Demo framework objective function
        info (Dict):          函数信息 {'lower', 'upper', 'dimension'} / Function info dict
        FEV_global (int):     函数评估总预算 / Total function evaluation budget
        seed (int, optional): 随机种子 / Random seed

    Returns:
        Tuple[List[float], float]:
            fitness_record: 100 个 best-so-far 检查点 / 100 best-so-far checkpoints
            elapsed:        运行时间（秒）/ Wall-clock time in seconds

    Raises:
        KeyError: 当 info 缺少必要字段时 / When info is missing required fields
    """
    for field_name in ('lower', 'upper', 'dimension'):
        if field_name not in info:
            raise KeyError(f"info 缺少必要字段 / Missing required field: '{field_name}'")
    if seed is not None:
        C.set_seed(seed)
    state = _allocate_arrays(FEV_global, info['lower'], info['upper'])
    initializePopulation(state.population, state.population_new,
                         POP_SIZE_MAX, DIMENSION, state.a_bound, state.b_bound)
    t0 = time.time()
    while state.FEV > 0:
        _run_outer_cycle(state, fun)
    while len(state.fitness_record) < 100:
        state.fitness_record.append(state.best_solution)
    elapsed = time.time() - t0
    logger.debug(f"Trial done: best={state.best_solution:.4e}, time={elapsed:.1f}s")
    return state.fitness_record, elapsed


def main() -> None:
    """
    命令行入口：运行 CC-SHADE-ML 并保存结果 / CLI entry point: run CC-SHADE-ML and save results.

    Usage:
        python run_ccshademl.py --id 1 --runs 25 --fev 3000000
    """
    import numpy as np
    parser = argparse.ArgumentParser(description="CC-SHADE-ML on LSGO CEC'2013")
    parser.add_argument("--id",   type=int, default=None,       help="函数ID 1-15 / Function ID 1-15")
    parser.add_argument("--runs", type=int, default=R_RUNS,     help="独立运行次数 / Number of independent runs")
    parser.add_argument("--fev",  type=int, default=FEV_GLOBAL, help="函数评估预算 / Function evaluation budget")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    benchmark = Benchmark()
    func_ids  = list(range(1, 16)) if args.id is None else [args.id]
    logger.info(f"CC-SHADE-ML starting: {func_ids}, runs={args.runs}, fev={args.fev:,}")
    print("=" * 60)
    print("CC-SHADE-ML  x  demo benchmark framework")
    print(f"  Functions: {func_ids}  |  runs={args.runs}  |  FEV={args.fev:,}")
    print("=" * 60)
    for fun_id in func_ids:
        fun  = benchmark.get_function(fun_id)
        info = benchmark.get_info(fun_id)
        print(f"\n▶ f{fun_id}  bounds=[{info['lower']}, {info['upper']}]")
        all_records: List[List[float]] = []
        all_times:   List[float]       = []
        for z in range(args.runs):
            record, elapsed = run_one_trial(fun, info, args.fev, seed=42 + z * 1000)
            all_records.append(record)
            all_times.append(elapsed)
            print(f"  Run {z+1:2d}/{args.runs}: best={record[-1]:.4e}  time={elapsed:.1f}s")
        finals = [r[-1] for r in all_records]
        _print_and_save_results(fun_id, finals, all_records, all_times, args)


if __name__ == "__main__":
    main()
