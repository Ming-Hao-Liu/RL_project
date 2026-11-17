"""Exploratory history-mechanism analysis of the unchanged formal benchmark.

No replanning, sample addition, or production changes. Uncertainty intervals are
pointwise, setting-stratified paired bootstrap intervals, not familywise claims.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.analyze_study import estimate, write_csv
from rl_retest.analysis import _rows
from rl_retest.experiments.records import read_jsonl

TOL = 1e-10  # numerical equality, not a practical-equivalence margin


def cell(r):
    return tuple((r[k] for k in ('k', 'horizon', 'n', 'p_c_current', 'cost')))


def summary(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple((r[k] for k in keys))].append(r)
    result = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        values = defaultdict(list)
        for r in items:
            values[cell(r)].append(r['advantage'])
        e = estimate(values)
        # Every main cell has 20 tasks. For exploratory strata, weight occupied
        # settings equally and expose sample/setting counts rather than pool them.
        def avg(field):
            cells = defaultdict(list)
            for r in items:
                cells[cell(r)].append(r[field])
            return float(np.mean([np.mean(x) for x in cells.values()]))
        result.append(dict(zip(keys,key))|e|dict(
            terminal_reward_gain=avg('terminal_reward_gain'),test_cost_saving=avg('test_cost_saving'),
            test_count_saving=avg('test_count_saving'),benefit_fraction=avg('benefits'),
            identical_policy_fraction=avg('identical_policy'),root_action_change_fraction=avg('root_action_changed'),
            correct_stop_fraction=avg('correct_stops'),other_stop_fraction=avg('other_stops'),
            descriptive_positive_case_gain=float(np.mean([r['advantage'] for r in items if r['benefits']])) if any(r['benefits'] for r in items) else 0.0))
    return result


def build(source, out):
    """Decompose fixed-policy differences into reward and testing cost terms."""
    source = source / 'main'
    out.mkdir(parents=True, exist_ok=True)
    manifest, records = _rows(source)
    tasks = {t['task_id']: t for t in read_jsonl(source / 'tasks.jsonl')}
    exact = defaultdict(dict)
    for r in records:
        if r['method'] == 'exact':
            assert r['memory'] not in exact[r['task_id']], 'duplicate exact condition'
            exact[r['task_id']][r['memory']] = r
    assert len(exact) == 3240
    rows = []
    for task_id, rs in exact.items():
        assert set(rs) == {'correct', 'reset', 'misread'}
        task = tasks[task_id]
        p = task['parameters']
        a = rs['correct']
        for memory in ('reset', 'misread'):
            b = rs[memory]
            gain = a['expected_utility'] - b['expected_utility']
            reward = manifest['config']['model']['reward'] * (a['success_probability'] - b['success_probability'])
            tests = b['expected_tests'] - a['expected_tests']
            saving = p['cost'] * tests
            assert abs(gain - reward - saving) < TOL
            assert gain >= -TOL
            root = lambda x: (x['policy']['action'], x['policy']['path'])
            rows.append(dict(task_id=task_id,**p,memory=memory,advantage=gain,
                terminal_reward_gain=reward,test_cost_saving=saving,test_count_saving=tests,
                benefits=int(gain>TOL),identical_policy=int(a['policy']==b['policy']),
                root_action_changed=int(root(a)!=root(b)),correct_stops=int(root(a)[0]=='stop'),
                other_stops=int(root(b)[0]=='stop'),
                high_A_prior_paths=sum(prior[2]==.7 for prior in task['priors']),
                any_high_A_prior=int(any(prior[2]==.7 for prior in task['priors'])),
                all_success_paths=sum(x==p['n'] for x in task['successes'][0]) if p['n'] else 0,
                any_all_success=int(p['n']>0 and any(x==p['n'] for x in task['successes'][0])),
                total_historical_successes=sum(task['successes'][0]),
                correct_utility=a['expected_utility'],other_utility=b['expected_utility']))
    # Exact diagnostic chosen to explain failed-history reinterpretation; it is
    # separate from the benchmark and carries no sampling confidence interval.
    from fractions import Fraction as F
    prior = (F(1, 5), F(3, 5), F(1, 5))
    old = (F(0), F(1, 10), F(9, 10))
    now = (F(0), F(9, 10), F(9, 10))
    weights = [b * (1 - p) ** 3 for b, p in zip(prior, old)]
    true = [w / sum(weights) for w in weights]
    wrong_weights = [b * (1 - p) ** 3 for b, p in zip(prior, now)]
    wrong = [w / sum(wrong_weights) for w in wrong_weights]
    mu = sum((b * p for b, p in zip(true, now)))
    wrong_mu = sum((b * p for b, p in zip(wrong, now)))
    q = F(1, 2)
    cost = F(1, 50)
    success_mu = sum((b * p * p for b, p in zip(true, now))) / mu
    failure_mu = sum((b * p * (1 - p) for b, p in zip(true, now))) / (1 - mu)
    test_value = -cost + mu * max(q, success_mu) + (1 - mu) * max(q, failure_mu)
    assert success_mu > q and failure_mu < q and (test_value > max(q, mu))
    wrong_s = sum((b * p * p for b, p in zip(wrong, now))) / wrong_mu
    wrong_f = sum((b * p * (1 - p) for b, p in zip(wrong, now))) / (1 - wrong_mu)
    wrong_test = -cost + wrong_mu * max(q, wrong_s) + (1 - wrong_mu) * max(q, wrong_f)
    assert wrong_test < q and wrong_mu < q
    diagnostic=dict(prior=[str(x) for x in prior],history='three failures at old (0,.1,.9)',current='(0,.9,.9)',horizon=1,cost=str(cost),
        correct_posterior=[str(x) for x in true],misread_posterior=[str(x) for x in wrong],
        correct_predictive=str(mu),misread_predictive=str(wrong_mu),correct_test_value=str(test_value),
        correct_policy='test; success -> candidate; failure -> fallback',misread_policy='stop fallback',
        misread_true_value=str(q),correct_minus_misread=str(test_value-q),type='exact illustrative diagnostic, not a sampled benchmark')
    (out / 'failure_diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
    write_csv(out / 'task_mechanisms.csv', rows)
    main = summary(rows, ['n', 'memory'])
    write_csv(out / 'overall.csv', main)
    cells = summary([r for r in rows if r['n'] == 3], ['memory', 'p_c_current', 'cost'])
    write_csv(out / 'capability_cost.csv', cells)
    hist = summary([r for r in rows if r['n'] == 3], ['memory', 'k', 'any_all_success'])
    write_csv(out / 'history_signal_strata.csv', hist)
    prior = summary([r for r in rows if r['n'] == 3], ['memory', 'k', 'any_high_A_prior'])
    write_csv(out / 'prior_composition_strata.csv', prior)
    # Cell-level exposure makes exploratory subgroup weighting inspectable.
    write_csv(out / 'full_cells.csv', summary(rows, ['memory', 'k', 'horizon', 'n', 'p_c_current', 'cost']))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey='row')
    for i, memory in enumerate(('reset', 'misread')):
        for j, cost in enumerate((0.005, 0.02, 0.1)):
            ax = axes[i, j]
            rs = sorted([r for r in cells if r['memory'] == memory and r['cost'] == cost], key=lambda r: r['p_c_current'])
            xs = [r['p_c_current'] for r in rs]
            for field, label, marker in [('mean', 'Total utility advantage', 'o'), ('terminal_reward_gain', 'Terminal reward gain', 's'), ('test_cost_saving', 'Saved test cost', '^')]:
                ax.plot(xs, [r[field] for r in rs], marker=marker, label=label)
            ax.fill_between(xs, [r['ci_low'] for r in rs], [r['ci_high'] for r in rs], alpha=0.15)
            ax.axhline(0, c='k', lw=0.6)
            ax.set_title(f'Correct - {memory}; cost={cost}')
            ax.set_xticks([0.1, 0.5, 0.9])
            ax.set_xlabel('Current capability probability')
            if j == 0:
                ax.set_ylabel('Expected utility units (R=1)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('History value decomposes into terminal reward and saved testing cost (n=3)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'history_decomposition.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, memory in zip(axes, ('reset', 'misread')):
        for pc in (0.1, 0.5, 0.9):
            rs = sorted([r for r in cells if r['memory'] == memory and r['p_c_current'] == pc], key=lambda r: r['cost'])
            ax.plot([r['cost'] for r in rs], [r['benefit_fraction'] for r in rs], marker='o', label=f'p_C={pc}')
        ax.set_xscale('log')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Correct - {memory}')
        ax.set_xlabel('Test cost')
        ax.set_ylabel('Fraction with utility gain > 1e-10')
        ax.legend()
    fig.suptitle('Where historical interpretation changes decision value (n=3)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'history_benefit_frequency.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    meta={'source':str(source),'source_tasks_sha256':manifest['tasks_sha256'],'source_config_sha256':manifest['config_sha256'],
        'source_planning_sha256':manifest['source_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'analysis':'post hoc exploratory reuse of fixed formal data; no new tasks',
        'bootstrap':{'samples':5000,'seed':20261129,'interval':'pointwise 95% percentile; setting-stratified paired tasks'},
        'numeric_equality_tolerance':TOL,'main_tasks':len(tasks),'paired_condition_rows':len(rows)}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    (out / 'README.md').write_text(
        '# 历史机制分析\n\n'
        '对主网格全部 3240 个任务作两种记忆比较，共 6480 对记录。'
        '完整参数切片使用设置等权的逐比较 95% 配对区间；先验和历史信号分层为探索性。\n\n'
        '- task_mechanisms.csv：逐任务效用 = 终止奖励改善 + 测试成本节省。\n'
        '- overall.csv、capability_cost.csv、full_cells.csv：总体、能力/成本、完整设置汇总。\n'
        '- history_signal_strata.csv、prior_composition_strata.csv：历史信号与先验的关联分层。\n'
        '- failure_diagnostic.json：三次失败的精确有理数案例，不是随机样本。\n'
        '- history_decomposition.pdf/png：总收益、终止奖励、测试成本；阴影为总收益区间。\n'
        '- history_benefit_frequency.pdf/png：收益大于 1e-10 的描述性频率。\n'
        '- manifest.json：来源摘要、分析脚本摘要和统计口径。\n\n'
        '1e-10 只区分浮点零，不是实际有意义阈值。分组关联不作独立因果解释。\n\n'
        '生成：`uv run python -m scripts.analyze_history --source data --out results/history`。\n'
    )
    print(json.dumps({'tasks': len(tasks), 'paired_rows': len(rows)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=Path('results/history'))
    a = p.parse_args()
    build(a.source, a.out)
