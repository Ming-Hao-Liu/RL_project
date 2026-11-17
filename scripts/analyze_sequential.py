"""Exploratory decomposition of the frozen benchmark and exact two-test diagnosis."""
import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.analysis import _rows
from rl_retest.model import ModelSpec, PlanningProblem, EvaluationContext
from rl_retest.planners.dynamic import solve_exact, solve_lookahead
from rl_retest.evaluation import evaluate
from rl_retest.experiments.records import policy_to_data
from scripts.analyze_study import estimate, write_csv

TOL = 1e-10


def summarize(rows, group, label, ci=True):
    values = np.array([r['gain'] for r in rows])
    positive = values > TOL
    cells = defaultdict(list)
    for r in rows:
        cells[tuple((r[k] for k in ('k', 'horizon', 'n', 'p_c_current', 'cost')))].append(r['gain'])
    est = estimate(cells) if ci else {}
    mean = float(values.mean())
    conditional = float(values[positive].mean()) if positive.any() else 0.0
    return dict(
        group=group,
        label=label,
        tasks=len(rows),
        positive=int(positive.sum()),
        frequency=float(positive.mean()),
        mean_gain=mean,
        conditional_gain=conditional,
        frequency_times_conditional=float(positive.mean()) * conditional,
        residual=mean - float(positive.mean()) * conditional,
        max_gain=float(values.max()),
        ci_low=est.get('ci_low', ''),
        ci_high=est.get('ci_high', ''),
        myopic_stop=sum((r['myopic_root'] == 'stop' for r in rows)),
        exact_stop=sum((r['exact_root'] == 'stop' for r in rows)),
        mean_exact_tests=float(np.mean([r['exact_tests'] for r in rows])),
        mean_myopic_tests=float(np.mean([r['myopic_tests'] for r in rows])),
        mean_terminal_reward_gain=float(np.mean([r['terminal_reward_gain'] for r in rows])),
        mean_extra_test_cost=float(np.mean([r['extra_expected_test_cost'] for r in rows])),
    )


def main(source, out):
    """Analyze all paired tasks and verify the fixed two-test cost boundary."""
    source = source / 'main'
    out.mkdir(parents=True, exist_ok=True)
    cstar = F(91, 11600)
    costs = [0.0, 0.001, 0.003, 0.005, 0.007, 0.0078, float(cstar), 0.0079, 0.008, 0.01, 0.02]
    protocol = {
        'classification': 'existing benchmark reanalysis is exploratory; analytic cost sweep is a diagnostic, not a new sampled confirmation',
        'question': 'Is the small mean DP-minus-rolling-myopic gain caused by rare gains or small conditional gains? What cost boundary permits two tests when one test has no benefit?',
        'source': str(source),
        'numerical_positive_threshold': TOL,
        'threshold_meaning': 'float error classification, not a minimum meaningful effect',
        'bootstrap_samples': 5000,
        'bootstrap_seed': 20261129,
        'fixed_diagnostic': {'prior': [0, 0.05, 0.95], 'probabilities': [0, 0.8, 0.05], 'fallback': 0.5, 'reward': 1, 'horizon': 2, 'costs': costs, 'analytic_boundary': str(cstar)},
        'stop_rule': 'analyze all 3240 existing pairs and all 11 declared cost points once; no sample extension',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    manifest, records = _rows(source)
    tasks = {r['task_id']: r for r in map(json.loads, (source / 'tasks.jsonl').read_text().splitlines())}
    selected = defaultdict(dict)
    for r in records:
        if r['memory'] == 'correct' and r['method'] in ('exact', 'myopic'):
            assert r['method'] not in selected[r['task_id']]
            selected[r['task_id']][r['method']] = r
    assert len(selected) == 3240 and all((set(v) == {'exact', 'myopic'} for v in selected.values()))
    rows = []
    for tid, v in selected.items():
        a, b = (v['exact'], v['myopic'])
        gain = a['expected_utility'] - b['expected_utility']
        assert gain >= -TOL
        ar, br = (a['policy']['action'], b['policy']['action'])
        kind = 'both_stop' if ar == br == 'stop' else 'myopic_stop_exact_test' if ar == 'test' and br == 'stop' else 'both_test_same_root' if ar == br == 'test' and a['policy']['path'] == b['policy']['path'] else 'both_test_different_root' if ar == br == 'test' else 'exact_stop_myopic_test'
        prior_count = sum((p[2] == 0.7 for p in tasks[tid]['priors']))
        rows.append(dict(
            task_id=tid,
            **a['parameters'],
            gain=gain,
            positive=gain > TOL,
            exact_root=ar,
            myopic_root=br,
            root_class=kind,
            prior_A_rich_count=prior_count,
            has_A_rich_prior=bool(prior_count),
            initial_stop=a['initial_stop'],
            perfect_information=a['perfect_information'],
            exact_tests=a['expected_tests'],
            myopic_tests=b['expected_tests'],
            terminal_reward_gain=a['success_probability'] - b['success_probability'],
            extra_expected_test_cost=a['parameters']['cost'] * (a['expected_tests'] - b['expected_tests']),
        ))
    write_csv(out / 'task_decomposition.csv', rows)
    summaries = [summarize(rows, 'all', 'all')]
    for field in ('horizon', 'cost', 'n', 'p_c_current', 'has_A_rich_prior', 'root_class'):
        for key in sorted({r[field] for r in rows}, key=str):
            subset = [r for r in rows if r[field] == key]
            # Outcome-conditioned groups are descriptive; no post-selection confidence claim.
            summaries.append(summarize(subset, field, str(key), ci=field in ('horizon', 'cost', 'n', 'p_c_current')))
    for b in (1, 2, 3):
        for c in (0.005, 0.02, 0.1):
            summaries.append(summarize([r for r in rows if r['horizon'] == b and r['cost'] == c], 'horizon_cost', f'B={b}, cost={c}'))
    assert max((abs(r['residual']) for r in summaries)) < TOL
    assert max((abs(r['gain'] - r['terminal_reward_gain'] + r['extra_expected_test_cost']) for r in rows)) < TOL
    write_csv(out / 'decomposition.csv', summaries)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    bs = [r for r in summaries if r['group'] == 'horizon']
    for ax, key, title in zip(axes, ('mean_gain', 'frequency', 'conditional_gain'), ('All-task mean gain', 'Fraction with gain > 1e-10', 'Mean gain among positive tasks')):
        errors = None
        if key == 'mean_gain':
            errors = [[r[key] - r['ci_low'] for r in bs],
                      [r['ci_high'] - r[key] for r in bs]]
            title += '\n(pointwise 95% paired bootstrap CI)'
        ax.bar([r['label'] for r in bs], [r[key] for r in bs], color='#356b9b', yerr=errors, capsize=4)
        ax.set_xlabel('Test horizon B')
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Expected utility (R=1)')
    axes[2].set_ylabel('Expected utility (R=1)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'gain_decomposition.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for b in (2, 3):
        rs = [r for r in summaries if r['group'] == 'horizon_cost' and r['label'].startswith(f'B={b},')]
        x = [0.005, 0.02, 0.1]
        axes[0].plot(x, [r['frequency'] for r in rs], '-o', label=f'B={b}')
        axes[1].plot(x, [r['mean_gain'] for r in rs], '-o', label=f'B={b}')
    for ax in axes:
        ax.set_xlabel('Test cost')
        ax.set_ylim(bottom=0)
        ax.legend()
    axes[0].set_ylabel('Fraction with gain > 1e-10')
    axes[1].set_ylabel('All-task mean utility gain')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'cost_decomposition.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    # Exact posterior table: type is drawn once, and likelihood accumulates over tests.
    posterior = []
    for obs in ('', 'F', 'S', 'FF', 'FS', 'SF', 'SS'):
        wc = F(1, 20)
        wa = F(19, 20)
        for o in obs:
            wc *= F(4, 5) if o == 'S' else F(1, 5)
            wa *= F(1, 20) if o == 'S' else F(19, 20)
        mass = wc + wa
        pc = wc / mass
        prediction = pc * F(4, 5) + (1 - pc) * F(1, 20)
        posterior.append(dict(
            observations=obs or 'root',
            sequence_probability=float(mass),
            posterior_C=float(pc),
            predictive_success=float(prediction),
            predictive_fraction=str(prediction),
            best_terminal='candidate' if prediction > F(1, 2) else 'fallback',
            optimal_action_at_cost_005='test' if obs in ('', 'S') else 'stop',
            reachable_under_optimal_policy=obs in ('', 'F', 'S', 'SF', 'SS'),
        ))
    write_csv(out / 'analytic_decision_table.csv', posterior)
    diagnostics = []
    for c in costs:
        problem = PlanningProblem(((0, 0.05, 0.95),), ModelSpec(1, c, 2, 0.5, (0, 0.8, 0.05), ()))
        exact = solve_exact(problem)
        myopic = solve_lookahead(problem, 1)
        analytic = max(0.5, 0.5 + float(F(273, 32000)) - float(F(87, 80)) * c)
        evaluation = evaluate(exact.policy, EvaluationContext(problem.belief, problem.spec))
        assert abs(exact.model_value - analytic) < TOL and abs(evaluation.expected_utility - analytic) < TOL and (abs(myopic.model_value - 0.5) < TOL)
        diagnostics.append(dict(
            cost=c,
            analytic_value=analytic,
            exact_value=exact.model_value,
            myopic_value=myopic.model_value,
            expected_tests=evaluation.expected_tests,
            policy=json.dumps(policy_to_data(exact.policy), sort_keys=True),
        ))
    write_csv(out / 'cost_diagnostic.csv', diagnostics)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(costs, [r['exact_value'] - 0.5 for r in diagnostics], '-o', label='Exact two-test gain')
    ax.plot(costs, [r['myopic_value'] - 0.5 for r in diagnostics], '--', label='Rolling myopic gain')
    ax.axvline(float(cstar), ls=':', c='k', label='Analytic threshold 91/11600')
    ax.set_xlabel('Test cost')
    ax.set_ylabel('Expected utility above fallback (R=1)')
    ax.set_ylim(bottom=-0.0005)
    ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'analytic_cost_boundary.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    best = max(rows, key=lambda r: r['gain'])
    sample = selected[best['task_id']]
    (out / 'representative_benchmark.json').write_text(json.dumps({
        'selection': 'maximum observed DP-minus-myopic gain; exploratory illustration, not representative sampling',
        'task': tasks[best['task_id']],
        'decomposition': best,
        'exact_record': sample['exact'],
        'myopic_record': sample['myopic'],
    }, indent=2))
    (out / 'provenance.json').write_text(json.dumps({
        'source_manifest': manifest,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_records': len(records),
        'paired_tasks': len(rows),
        'checks': {'decomposition_residual_lt_1e_10': True, 'reward_minus_cost_identity': True, 'analytic_DP_evaluation_agreement': True},
    }, indent=2))
    (out / 'README.md').write_text(
        '# 序贯测试机制分析\n\n'
        '主网格全部 3240 对 exact / rolling-myopic 策略。正收益阈值 1e-10 仅排除舍入误差；'
        '条件正收益均值描述已观察子群，不替代总体均值。\n\n'
        '- task_decomposition.csv：逐任务差值、根动作、终止奖励和测试成本。\n'
        '- decomposition.csv：总体均值 = 获益频率 × 条件平均收益，以及预定参数切片。\n'
        '- gain_decomposition.pdf/png：总体差值（逐比较 95% 配对区间）、频率、条件差值。\n'
        '- cost_decomposition.pdf/png：成本分组的频率与总体差。\n'
        '- analytic_decision_table.csv：固定两类型例子的后验与最优分支。\n'
        '- cost_diagnostic.csv、analytic_cost_boundary.pdf/png：11 个声明成本点与解析阈值 91/11600。\n'
        '- representative_benchmark.json：最大观察收益的示例，属于探索性选择。\n'
        '- protocol.json、provenance.json：诊断范围、来源和恒等式验证。\n\n'
        '解析诊断不附采样区间；根动作和按结果筛选的子群不作确认性推断。\n\n'
        '生成：`uv run python -m scripts.analyze_sequential --source data --out results/sequential`。\n'
    )
    print(json.dumps({'overall': summaries[0], 'by_horizon': bs}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=Path('results/sequential'))
    a = p.parse_args()
    main(a.source, a.out)
