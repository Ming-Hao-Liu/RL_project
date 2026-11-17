"""Independently evaluate fixed policies and audit successor-update accounting."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

from rl_retest.analysis import _rows
from rl_retest.experiments.records import read_jsonl
from rl_retest.model import ModelSpec, PlanningProblem
from rl_retest.planners import common
from rl_retest.planners.blinkered import solve_blinkered
from rl_retest.planners.bounded import solve_bounded
from rl_retest.planners.dynamic import solve_exact, solve_iterative, solve_lookahead
from tests.oracle import fixed_policy, joint


def rational_value(policy, task, model):
    """Use joint types and rational likelihoods, without production updating."""
    historical = (model['p_i'], model['p_c_old'], model['p_a'])
    current = (model['p_i'], task['parameters']['p_c_current'], model['p_a'])
    distribution = joint(
        task['priors'], [historical], task['successes'], task['failures']
    )
    return fixed_policy(
        policy, distribution, current, model['fallback'], model['reward'],
        task['parameters']['cost'],
    )


def review_policies(tasks, records, model):
    """Check all non-alias exact/bounded policies in the fixed 18 settings."""
    selected = {
        task_id: task for task_id, task in tasks.items()
        if task['parameters']['k'] == 3
        and task['parameters']['index'] == 0
        and task['parameters']['horizon'] == 3
    }
    results = []
    # Preserve a stable audit order independent of file-system traversal order.
    for record in sorted(records, key=lambda r: r['job_id']):
        if (record['task_id'] not in selected or 'alias_of' in record
                or record['method'] not in ('exact', 'bounded')):
            continue
        task = selected[record['task_id']]
        value = float(rational_value(record['policy'], task, model))
        error = abs(value - record['expected_utility'])
        assert error < 1e-10
        results.append({
            'task_id': task['task_id'],
            'job_id': record['job_id'],
            'method': record['method'],
            'memory': record['memory'],
            'rational_value': value,
            'saved_value': record['expected_utility'],
            'error': error,
        })
    assert len(selected) == 18 and len(results) == 51
    checks = {
        'independent_policy_evaluations': len(results),
        'independent_tasks': len(selected),
        'max_value_error': max(r['error'] for r in results),
        'reference_method': 'rational joint posterior + fixed-type policy evaluation; no production posterior or evaluate',
    }
    return checks, results


def review_accounting():
    """Count actual Bayesian calls independently of each planner's counters."""
    problem = PlanningProblem(
        ((.2, .6, .2), (.1, .2, .7), (.6, .3, .1)),
        ModelSpec(1, .005, 3, .5, (0, .5, .9), ()),
    )
    solvers = (
        ('exact', solve_exact),
        ('myopic', lambda p: solve_lookahead(p, 1)),
        ('depth2', lambda p: solve_lookahead(p, 2)),
        ('blinkered', solve_blinkered),
        ('iterative100', lambda p: solve_iterative(p, 100)),
        ('bounded100', lambda p: solve_bounded(p, 100, 0)),
    )
    original = common.update
    counts = []
    for name, solver in solvers:
        observed = 0

        def counted(*args, **kwargs):
            nonlocal observed
            observed += 1
            return original(*args, **kwargs)

        common.update = counted
        try:
            result = solver(problem)
        finally:
            common.update = original
        assert observed == result.evaluations
        counts.append({
            'method': name,
            'observed_updates': observed,
            'reported_updates': result.evaluations,
            'cache_hits': result.cache_hits,
            'status': result.status,
        })
    return counts


def review_tolerance(tasks, records, model, diagnostics):
    """Check the full diagnostic selection and one predetermined rational case."""
    with (diagnostics / 'tolerance_diagnostic.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    policies = {
        r['task_id']: r['policy']
        for r in read_jsonl(diagnostics / 'tolerance_policies.jsonl')
    }
    refs = {
        r['task_id']: r for r in records
        if r['method'] == 'exact' and r['memory'] == 'correct'
    }
    selected = {
        r['task_id'] for r in records if r['method'] == 'bounded'
        and refs[r['task_id']]['model_value'] - r['model_value'] > 1e-10
    }
    assert len(rows) == len(selected) == len(policies) == 199
    assert {r['task_id'] for r in rows} == selected == set(policies)
    for row in rows:
        assert row['old_status'] == 'tolerance' and row['new_status'] == 'complete'
        assert abs(float(row['new_value']) - refs[row['task_id']]['model_value']) < 1e-10
        assert float(row['new_width']) == 0
        assert int(row['new_evaluations']) <= 10000
    task_id = min(t for t in selected if tasks[t]['parameters']['k'] == 3)
    row = next(r for r in rows if r['task_id'] == task_id)
    value = rational_value(policies[task_id], tasks[task_id], model)
    error = abs(float(value) - float(row['new_value']))
    assert error < 1e-10
    return {
        'tasks': len(rows),
        'all_old_tolerance_new_exact': True,
        'selected_task_id': task_id,
        'parameters': tasks[task_id]['parameters'],
        'rational_new_value': str(value),
        'float_new_value': float(value),
        'stored_value': float(row['new_value']),
        'old_value': float(row['old_value']),
        'error': error,
    }


def main(source: Path, out: Path, diagnostics: Path):
    manifest, records = _rows(source / 'main')
    tasks = {r['task_id']: r for r in read_jsonl(source / 'main/tasks.jsonl')}
    model = manifest['config']['model']
    checks, values = review_policies(tasks, records, model)
    accounting = review_accounting()
    tolerance = review_tolerance(tasks, records, model, diagnostics)
    out.mkdir(parents=True, exist_ok=True)
    for filename, content in (
        ('checks.json', checks),
        ('independent_policy_values.json', values),
        ('transition_accounting.json', accounting),
        ('tolerance_review.json', tolerance),
    ):
        (out / filename).write_text(json.dumps(content, indent=2))
    (out / 'provenance.json').write_text(json.dumps({
        'source': str(source),
        'diagnostics': str(diagnostics),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': manifest['source_sha256'],
    }, indent=2))
    (out / 'README.md').write_text(
        '# 验证记录\n\n'
        '- checks.json、independent_policy_values.json：固定 18 个设置中 51 条策略的独立有理数评价。\n'
        '- transition_accounting.json：六种规划方法的实际更新调用数与计数器比较。\n'
        '- tolerance_review.json：全部 199 条容差诊断记录及一条预定策略的有理数复核。\n'
        '- provenance.json：数据来源、诊断输入和验证脚本摘要。\n\n'
        '复核：`uv run python -m scripts.review_evidence --source data '
        '--diagnostics results/search --out results/validation`。\n'
    )
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('results/validation'))
    parser.add_argument('--diagnostics', type=Path, default=Path('results/search'))
    args = parser.parse_args()
    main(args.source, args.out, args.diagnostics)
