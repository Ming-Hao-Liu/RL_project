"""Exploratory quality/cost/certificate decomposition of the fixed benchmark."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.analysis import _rows
from rl_retest.experiments.records import read_jsonl, policy_to_data
from rl_retest.experiments.tasks import Task, contexts
from rl_retest.planners.common import TransitionCache
from rl_retest.planners.dynamic import _Bellman
from scripts.analyze_study import write_csv, estimate


def save(fig, name, out):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / (name + '.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)

def main(source, out):
    """Separate realized policy quality, natural cost and model certificates."""
    out.mkdir(parents=True, exist_ok=True)
    main_manifest, rows = _rows(source / 'main')
    exact = {r['task_id']: r for r in rows if r['method'] == 'exact' and r['memory'] == 'correct'}
    quality = []
    for h in (1, 2, 3):
        for method in ('myopic', 'depth2', 'blinkered', 'bounded', 'iterative'):
            rr = [r for r in rows if r['method'] == method and r['memory'] == 'correct' and (r['parameters']['horizon'] == h)]
            losses = np.array([exact[r['task_id']]['expected_utility'] - r['expected_utility'] for r in rr])
            assert losses.min() > -1e-10
            widths = [r['upper'] - r['lower'] for r in rr if r['upper'] is not None]
            quality.append(dict(
                horizon=h,
                method=method,
                tasks=len(rr),
                exact_value_fraction=float(np.mean(losses <= 1e-10)),
                mean_regret=float(losses.mean()),
                max_regret=float(losses.max()),
                mean_width=float(np.mean(widths)) if widths else '',
                exact_but_positive_width=sum((loss <= 1e-10 and r['upper'] is not None and (r['upper'] - r['lower'] > 1e-10) for loss, r in zip(losses, rr))),
            ))
    write_csv(out / 'main_quality.csv', quality)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for method in ('myopic', 'depth2', 'blinkered', 'bounded', 'iterative'):
        rs = [r for r in quality if r['method'] == method]
        axs[0].plot([r['horizon'] for r in rs], [r['exact_value_fraction'] for r in rs], '-o', label=method)
        axs[1].plot([r['horizon'] for r in rs], [r['mean_regret'] for r in rs], '-o', label=method)
    axs[0].set_ylim(0, 1.02)
    axs[0].set_ylabel('Fraction attaining exact value (tolerance 1e-10)')
    axs[1].set_ylabel('Mean exact regret')
    axs[1].legend()
    for ax in axs:
        ax.set_xlabel('Test horizon B')
        ax.set_xticks([1, 2, 3])
    save(fig, 'shallow_sufficiency', out)
    manifest, large = _rows(source / 'large')
    records = []
    summary = []
    convergence = []
    for n in (0, 3):
        selected = [r for r in large if r['parameters']['n'] == n]
        for method in ('stop', 'myopic', 'blinkered', 'bounded', 'iterative'):
            for budget in (100, 1000, 10000) if method in ('bounded', 'iterative') else (None,):
                rr = [r for r in selected if r['method'] == method and r.get('budget') == budget and (method != 'bounded' or r['epsilon'] == 0.01)]
                summary.append(dict(
                    n=n,
                    method=method,
                    budget=budget,
                    tasks=len(rr),
                    mean_utility=np.mean([r['expected_utility'] for r in rr]),
                    mean_evaluations=np.mean([r['evaluations'] for r in rr]),
                    median_seconds=np.median([r['planning_seconds'] for r in rr]),
                    p90_seconds=np.quantile([r['planning_seconds'] for r in rr], 0.9),
                    mean_scans=np.mean([r['action_scans'] for r in rr]),
                    mean_width=np.mean([r['upper'] - r['lower'] for r in rr]) if method == 'bounded' else '',
                ))
    write_csv(out / 'natural_cost.csv', summary)
    policies = {}
    for r in large:
        if r['method'] in ('myopic', 'blinkered'):
            policies.setdefault(r['task_id'], {})[r['method']] = r
    for r in large:
        if r['method'] != 'bounded' or r['epsilon'] != 0.01:
            continue
        baseline = policies[r['task_id']]['myopic']
        assert r['upper'] + 1e-10 >= baseline['model_value']
        records.append(dict(
            task_id=r['task_id'],
            **r['parameters'],
            budget=r['budget'],
            lower=r['lower'],
            upper=r['upper'],
            initial_upper=max(r['initial_stop'], r['perfect_information'] - r['parameters']['cost']),
            baseline_value=baseline['model_value'],
            baseline_gap_upper=r['upper'] - baseline['model_value'],
            search_gap_upper=r['upper'] - r['lower'],
            baseline_evaluations=baseline['evaluations'],
            search_evaluations=r['evaluations'],
        ))
    write_csv(out / 'certificate_tasks.csv', records)
    for n in (0, 3):
        for m in (100, 1000, 10000):
            rr = [r for r in records if r['n'] == n and r['budget'] == m]
            convergence.append(dict(
                n=n,
                budget=m,
                mean_lower=np.mean([r['lower'] for r in rr]),
                mean_upper=np.mean([r['upper'] for r in rr]),
                mean_baseline=np.mean([r['baseline_value'] for r in rr]),
                mean_initial_upper=np.mean([r['initial_upper'] for r in rr]),
                mean_baseline_gap_upper=np.mean([r['baseline_gap_upper'] for r in rr]),
            ))
    write_csv(out / 'certificate_convergence.csv', convergence)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for n, ax in zip((0, 3), axs):
        rr = [r for r in convergence if r['n'] == n]
        x = [r['budget'] for r in rr]
        for key, label in [('mean_lower', 'Search policy L'), ('mean_upper', 'Search upper U'), ('mean_baseline', 'Myopic feasible value'), ('mean_initial_upper', 'Initial upper')]:
            ax.plot(x, [r[key] for r in rr], '-o', label=label)
        ax.set_xscale('log')
        ax.set_xlabel('Successor-belief budget M')
        ax.set_ylabel('Mean utility / model bound')
        ax.set_title(f'n={n}')
    axs[1].text(0.04, 0.05, 'All four curves coincide: stop certified', transform=axs[1].transAxes, fontsize=9)
    axs[1].legend()
    save(fig, 'certificate_decomposition', out)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for method in ('stop', 'myopic', 'blinkered', 'bounded', 'iterative'):
        rr = [r for r in summary if r['n'] == 0 and r['method'] == method]
        axs[0].plot([max(1, r['mean_evaluations']) for r in rr], [r['mean_utility'] for r in rr], '-o', label=method)
        axs[1].plot([r['median_seconds'] for r in rr], [r['mean_utility'] for r in rr], '-o', label=method)
    for ax in axs:
        ax.set_xscale('log')
        ax.set_ylabel('Mean utility, n=0')
    axs[0].set_xlabel('Mean successor computations (STOP shown at 1)')
    axs[1].set_xlabel('Median observed single-worker seconds')
    axs[1].legend()
    save(fig, 'quality_cost', out)
    # Predetermined implementation diagnostic: first K=20/40, B=6, n=0 tasks.
    protocol = {
        'question': 'Does full-depth completion explain iterative budget plateaus?',
        'selection': 'index=0,K in [20,40],B=6,n=0 from frozen large tasks',
        'depths': [1, 2],
        'expected_computations': 'd1=2K; d2=2K+(2K)^2 for strictly positive outcome branches',
        'classification': 'deterministic implementation diagnostic, no population inference',
    }
    (out / 'diagnostic_protocol.json').write_text(json.dumps(protocol, indent=2))
    diagnostic = []
    for data in read_jsonl(source / 'large/tasks.jsonl'):
        p = data['parameters']
        if p['index'] != 0 or p['horizon'] != 6 or p['n'] != 0:
            continue
        task = Task.from_data(data)
        problem, _ = contexts(task, manifest['config'])
        cache = TransitionCache(problem)
        bellman = _Bellman(problem, cache)
        for depth in (1, 2):
            val, policy = bellman.solve((0,) * (2 * p['k']), problem.belief, depth)
            expected = 2 * p['k'] + (2 * p['k']) ** 2 if depth == 2 else 2 * p['k']
            assert cache.stats.evaluations == expected
            saved = [r for r in large if r['task_id'] == task.task_id and r['method'] == 'iterative' and (r['budget'] == (1000 if depth == 1 else 10000))][0]
            assert policy_to_data(policy) == saved['policy']
            diagnostic.append(dict(k=p['k'], depth=depth, computations=cache.stats.evaluations, model_value=val, saved_policy_match=True))
    write_csv(out / 'depth_completion.csv', diagnostic)
    plateau = []
    for n in (0, 3):
        ids = {r['task_id'] for r in large if r['parameters']['n'] == n}
        equal = 0
        for tid in ids:
            rr = {r['budget']: r for r in large if r['task_id'] == tid and r['method'] == 'iterative'}
            equal += rr[100]['policy'] == rr[1000]['policy']
        plateau.append(dict(n=n, tasks=len(ids), identical_policy_M100_M1000=equal))
    write_csv(out / 'iterative_plateau.csv', plateau)
    _, stress = _rows(source / 'stress')
    nom = {r['task_id']: r for r in stress if r.get('scenario') == 'nominal'}
    stability = []
    for n in (0, 3):
        for pc in (0.4, 0.6):
            rr = [r for r in stress if r.get('scenario') == 'probability_shift' and r['method'] == 'exact' and (r['parameters']['n'] == n) and (r['assumed_current'] == pc)]
            stability.append(dict(
                n=n,
                assumed_current=pc,
                tasks=len(rr),
                identical_policy=sum((r['policy'] == nom[r['task_id']]['policy'] for r in rr)),
                same_value=sum((abs(r['expected_utility'] - nom[r['task_id']]['expected_utility']) < 1e-10 for r in rr)),
            ))
    write_csv(out / 'stress_policy_stability.csv', stability)
    (out / 'provenance.json').write_text(json.dumps({
        'source': str(source),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': main_manifest['source_sha256'],
        'classification': 'exploratory reanalysis and fixed depth-count diagnostic',
    }, indent=2))
    (out / 'README.md').write_text(
        '# 规划质量、成本与证书\n\n'
        '固定正式任务的探索性机制分析。M 计后继信念更新，不是全部 CPU 运算；'
        '耗时是单工作进程观测值，包含输入初始化。\n\n'
        '- main_quality.csv、shallow_sufficiency.pdf/png：主网格精确值比例和实际 regret。\n'
        '- natural_cost.csv、quality_cost.pdf/png：大规模策略值、自然更新成本及观测耗时。\n'
        '- certificate_tasks.csv、certificate_convergence.csv、certificate_decomposition.pdf/png：'
        '可执行下界、乐观上界与 myopic 值，不能将界宽当真实 regret。\n'
        '- depth_completion.csv、iterative_plateau.csv：当前概率表的深度完成门槛及策略平台。\n'
        '- stress_policy_stability.csv：指定概率误设条件下的策略一致性。\n'
        '- diagnostic_protocol.json：深度计数诊断任务的确定规则。\n'
        '- tolerance_protocol.json、tolerance_diagnostic.csv、tolerance_policies.jsonl：'
        '由 diagnose_tolerance 生成的全部非精确任务容差干预及完整策略。\n\n'
        '深度 2 的计数 2K+(2K)^2 适用于当前根及第一层后两分支均为正概率的模型。'
        '大规模真实最优未知；U−J(myopic) 需要额外搜索工作，属于事后上界。\n\n'
        '生成：`uv run python -m scripts.analyze_search --source data --out results/search`。\n'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('results/search'))
    args = parser.parse_args()
    main(args.source, args.out)
