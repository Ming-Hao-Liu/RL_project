"""Generate benchmark tables and figures from fixed policies; never replan."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.analysis import _rows, _cell


def estimate(values, seed=20261129):
    """Equal-weight settings; resample paired task differences within each setting."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(5000)
    means = []
    for key in sorted(values, key=str):
        a = np.asarray(values[key], dtype=float)
        means.append(float(a.mean()))
        draws += a[rng.integers(0, len(a), (5000, len(a)))].mean(axis=1)
    draws /= len(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(
        mean=float(np.mean(means)),
        ci_low=float(low),
        ci_high=float(high),
        tasks=sum(map(len, values.values())),
        settings=len(values),
    )


def paired(left, right):
    """Require exactly one observation per task on each side of a comparison."""
    a = {r['task_id']: r for r in left}
    b = {r['task_id']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys() or (not a):
        raise ValueError('duplicate or incomplete paired observations')
    cells = defaultdict(list)
    for key, r in a.items():
        cells[_cell(r)].append(r['expected_utility'] - b[key]['expected_utility'])
    return estimate(cells)


def write_csv(path, rows):
    """Keep field order and optional columns across heterogeneous record groups."""
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def interval_plot(path, rows, title):
    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.35)))
    for i, r in enumerate(rows):
        ax.plot([r['ci_low'], r['ci_high']], [i, i], color='C0')
        ax.plot(r['mean'], i, 'o', color='C0')
    ax.set_yticks(range(len(rows)), [r['label'] for r in rows])
    ax.axvline(0, c='k', lw=0.6)
    ax.set_xlabel(title + ' (95% paired bootstrap CI)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)


def build(source, out):
    """Summarize all main, large and stress records without copying raw inputs."""
    out.mkdir(parents=True, exist_ok=True)
    tables = out / 'tables'
    figures = out / 'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    data = {}
    coverage = []
    for suite in ('main', 'large', 'stress'):
        root = source / suite
        manifest, rows = _rows(root)
        verification = json.loads((root / 'verification.json').read_text())
        if not verification['valid']:
            raise ValueError('verification failed')
        data[suite] = rows
        coverage.append(dict(
            suite=suite,
            tasks=manifest['task_count'],
            jobs=manifest['job_count'],
            records=len(rows),
            certificate_checks=verification['certificate_checks'],
            valid=True,
            statuses=json.dumps(dict(Counter((r['status'] for r in rows)))),
            source_sha256=manifest['source_sha256'],
            config_sha256=manifest['config_sha256'],
        ))
    write_csv(tables / 'coverage.csv', coverage)
    main = data['main']
    large = data['large']
    stress = data['stress']
    def select(rows, **kw):
        return [r for r in rows if all(((r['parameters'].get(k) if k in r['parameters'] else r.get(k)) == v for k, v in kw.items()))]
    memory = []
    for n in (0, 3):
        for other in ('reset', 'misread'):
            memory.append(dict(
                label=f'n={n}: correct - {other}',
                **paired(select(main, method='exact', memory='correct', n=n), select(main, method='exact', memory=other, n=n)),
            ))
    write_csv(tables / 'memory.csv', memory)
    multistep = []
    for b in (1, 2, 3):
        multistep.append(dict(
            label=f'B={b}: exact - myopic',
            **paired(select(main, method='exact', memory='correct', horizon=b), select(main, method='myopic', memory='correct', horizon=b)),
        ))
    write_csv(tables / 'multistep.csv', multistep)
    interval_plot(figures / 'multistep', multistep, 'Exact minus myopic utility')
    budget = []
    strata = []
    for n in (0, 3):
        for epsilon in (0.01, 0.05):
            for m in (100, 1000, 10000):
                a = select(large, method='bounded', n=n, budget=m, epsilon=epsilon)
                b = select(large, method='iterative', n=n, budget=m)
                for layer in ('all', 'needs_search'):
                    aa = a if layer == 'all' else [r for r in a if max(0, r['perfect_information'] - r['parameters']['cost'] - r['initial_stop']) > epsilon]
                    ids = {r['task_id'] for r in aa}
                    bb = [r for r in b if r['task_id'] in ids]
                    if aa:
                        budget.append(dict(
                            label=f'n={n}, M={m}, e={epsilon}, {layer}',
                            n=n,
                            budget=m,
                            epsilon=epsilon,
                            stratum=layer,
                            **paired(aa, bb),
                        ))
            aa = select(large, method='bounded', n=n, budget=10000, epsilon=epsilon)
            stop = sum((r['perfect_information'] - r['initial_stop'] <= r['parameters']['cost'] for r in aa))
            tol = sum((0 < r['perfect_information'] - r['parameters']['cost'] - r['initial_stop'] <= epsilon for r in aa))
            strata.append(dict(
                n=n,
                epsilon=epsilon,
                tasks=len(aa),
                certified_stop=stop,
                initial_tolerance_only=tol,
                needs_search=len(aa) - stop - tol,
                mean_information_value=float(np.mean([r['perfect_information'] - r['initial_stop'] for r in aa])),
            ))
    write_csv(tables / 'budget_pairs.csv', budget)
    write_csv(tables / 'stop_strata.csv', strata)
    interval_plot(figures / 'budget_pairs', [r for r in budget if r['epsilon'] == 0.01 and r['stratum'] == 'all'], 'Bounded minus iterative utility')
    natural = []
    for n in (0, 3):
        for baseline in ('myopic', 'blinkered'):
            natural.append(dict(
                label=f'n={n}: bounded M=10000 - {baseline} (natural cost)',
                **paired(select(large, method='bounded', n=n, budget=10000, epsilon=0.01), select(large, method=baseline, n=n)),
            ))
    write_csv(tables / 'natural_baselines.csv', natural)
    interval_plot(figures / 'natural_baselines', natural, 'Utility difference; baseline costs are unmatched')
    history_pairs = []
    for method in ('stop', 'myopic', 'blinkered', 'bounded', 'iterative'):
        rows = select(large, method=method)
        if method in ('bounded', 'iterative'):
            rows = select(rows, budget=10000)
        if method == 'bounded':
            rows = select(rows, epsilon=0.01)
        keyed = {}
        for r in rows:
            p = r['parameters']
            key = (p['k'], p['horizon'], p['index'])
            keyed.setdefault(key, {})[p['n']] = r
        cells = defaultdict(list)
        for key, rs in keyed.items():
            assert set(rs) == {0, 3}
            cells[key[:2]].append(rs[3]['expected_utility'] - rs[0]['expected_utility'])
        history_pairs.append(dict(label=method + ': n=3 minus n=0', **estimate(cells)))
    write_csv(tables / 'large_history_pairs.csv', history_pairs)
    mismatch = []
    resampling = []
    for n in (0, 3):
        truth = {r['task_id']: r for r in select(stress, scenario='nominal', n=n)}
        for pc in (0.4, 0.6):
            for method in ('exact', 'bounded'):
                rows = select(stress, scenario='probability_shift', n=n, assumed_current=pc, method=method)
                for metric in ('true_regret', 'absolute_model_error', 'certificate_width'):
                    cells = defaultdict(list)
                    for r in rows:
                        v = truth[r['task_id']]['expected_utility'] - r['expected_utility'] if metric == 'true_regret' else abs(r['model_value'] - r['expected_utility']) if metric == 'absolute_model_error' else r['upper'] - r['lower']
                        cells[_cell(r)].append(v)
                    mismatch.append(dict(label=f'n={n}, p={pc}, {method}, {metric}', metric=metric, **estimate(cells)))
        a = select(stress, scenario='type_resampling', n=n, method='exact', memory='correct')
        for method, mem in (('exact', 'reset'), ('bounded', 'correct')):
            b = select(stress, scenario='type_resampling', n=n, method=method, memory=mem)
            resampling.append(dict(label=f'n={n}: nominal DP - {method}/{mem}', **paired(a, b)))
    write_csv(tables / 'probability_shift.csv', mismatch)
    write_csv(tables / 'resampling.csv', resampling)
    interval_plot(figures / 'probability_shift', [r for r in mismatch if r['metric'] == 'true_regret'], 'True regret (correct-model optimum)')
    interval_plot(figures / 'model_error', [r for r in mismatch if r['metric'] == 'absolute_model_error'], 'Absolute model-value error')
    interval_plot(figures / 'resampling', resampling, 'True utility difference after one-path resampling')
    # Counterexample figure, values independently checked by tests.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    nodes = [(0.08, 0.65, 'Test 1'), (0.4, 0.85, 'Failure: fallback'), (0.4, 0.4, 'Success: test 2'), (0.85, 0.6, 'Failure: fallback'), (0.85, 0.2, 'Success: candidate')]
    labels = [ax.text(x, y, label, ha='center', va='center', bbox=dict(boxstyle='round', fc='#eef4fa'), zorder=3) for x, y, label in nodes]
    fig.canvas.draw()
    for a, b in ((0, 1), (0, 2), (2, 3), (2, 4)):
        ax.annotate('', xy=nodes[b][:2], xytext=nodes[a][:2], arrowprops=dict(
            arrowstyle='->',
            patchA=labels[a].get_bbox_patch(),
            patchB=labels[b].get_bbox_patch(),
            shrinkA=5,
            shrinkB=5,
        ), zorder=1)
    ax.set_title('Sequential testing: V1 = 0.5; V2 = 0.50309375; cost = 0.005')
    for ext in ('png', 'pdf'):
        fig.savefig(figures / ('sequential_tree.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)
    checks = {}
    for name, rows in [('no_history', select(main, n=0)), ('unchanged_capability', select(main, n=3, p_c_current=0.1))]:
        memories = ('reset', 'misread') if name == 'no_history' else ('misread',)
        for mem in memories:
            a = {r['task_id']: r for r in select(rows, method='exact', memory='correct')}
            b = {r['task_id']: r for r in select(rows, method='exact', memory=mem)}
            error = max((abs(a[k]['expected_utility'] - b[k]['expected_utility']) for k in a))
            assert error < 1e-10
            checks[name + '/' + mem] = error
    for r in select(stress, n=0, scenario='type_resampling'):
        assert abs(r['model_value'] - r['expected_utility']) < 1e-10
    checks['resampling_n0'] = True
    (out / 'negative_controls.json').write_text(json.dumps(checks, indent=2))
    (out / 'README.md').write_text(
        '# 正式基准统计\n\n'
        '来源：三个正式数据集的完整记录。参数设置等权，设置内配对 bootstrap 5000 次；'
        '区间为逐比较 95%，不表示所有比较的联合覆盖。\n\n'
        '- tables/coverage.csv：任务、作业、记录和验证覆盖。\n'
        '- tables/memory.csv：正确历史相对 reset / misread 的真实效用差。\n'
        '- tables/multistep.csv：精确 DP 相对滚动 myopic 的差。\n'
        '- tables/budget_pairs.csv、natural_baselines.csv：同更新预算与自然成本比较。\n'
        '- tables/stop_strata.csv、large_history_pairs.csv：停止分层及共享先验配对。\n'
        '- tables/probability_shift.csv、resampling.csv：概率表误设及类型重采样。\n'
        '- figures/：各比较的 PDF / PNG，及固定两步解析策略树。\n'
        '- negative_controls.json：无历史、能力不变及无历史重采样的等价检查。\n\n'
        '生成：`uv run python -m scripts.analyze_study --source data --out results/study`。\n'
    )
    (out / 'provenance.json').write_text(json.dumps({
        'source': str(source),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'suites': coverage,
        'bootstrap_samples': 5000,
        'bootstrap_seed': 20261129,
        'interval': 'pointwise 95% setting-stratified paired bootstrap',
    }, indent=2))
    return coverage

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=Path('results/study'))
    a = p.parse_args()
    build(a.source, a.out)
