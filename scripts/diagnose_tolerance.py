"""Evaluate epsilon-zero search on every nonoptimal formal main-search output.

This is a paired mechanism diagnostic on existing tasks, not a new population
sample. The fixed protocol is written before any diagnostic policy is solved.
"""
import argparse
import json
import hashlib
import time
from pathlib import Path

from rl_retest.analysis import _rows
from rl_retest.evaluation import evaluate
from rl_retest.experiments.records import policy_to_data, read_jsonl
from rl_retest.experiments.tasks import Task, contexts
from rl_retest.planners.bounded import solve_bounded
from scripts.analyze_study import write_csv


def main(source: Path, out: Path):
    """Change only stopping tolerance; preserve tasks, model and update budget."""
    root = source / 'main'
    manifest, rows = _rows(root)
    refs = {
        r['task_id']: r for r in rows
        if r['method'] == 'exact' and r['memory'] == 'correct'
    }
    selected = [
        r for r in rows if r['method'] == 'bounded'
        and refs[r['task_id']]['model_value'] - r['model_value'] > 1e-10
    ]
    tasks = {
        r['task_id']: Task.from_data(r)
        for r in read_jsonl(root / 'tasks.jsonl')
    }
    protocol = {
        'classification': 'exploratory paired intervention on existing tasks, no new population inference',
        'question': 'Are observed small-task search losses explained by positive stopping tolerance?',
        'selection': 'ALL main bounded records with exact-minus-search >1e-10',
        'count': len(selected),
        'epsilon_old': .01,
        'epsilon_new': 0,
        'budget_unchanged': 10000,
        'no_additional_samples': True,
        'task_ids': [r['task_id'] for r in selected],
        'source_sha256': manifest['source_sha256'],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.mkdir(exist_ok=True, parents=True)
    (out / 'tolerance_protocol.json').write_text(json.dumps(protocol, indent=2))
    results, policies = [], []
    for original in selected:
        problem, truth = contexts(tasks[original['task_id']], manifest['config'])
        start = time.perf_counter()
        plan = solve_bounded(problem, 10000, 0)
        elapsed = time.perf_counter() - start
        value = evaluate(plan.policy, truth).expected_utility
        optimum = refs[original['task_id']]['model_value']
        assert abs(value - plan.lower) <= 1e-10
        assert plan.lower <= optimum + 1e-10
        assert optimum <= plan.upper + 1e-10
        assert plan.evaluations <= 10000
        results.append({
            'task_id': original['task_id'],
            'horizon': original['parameters']['horizon'],
            'old_status': original['status'],
            'new_status': plan.status,
            'old_value': original['model_value'],
            'new_value': value,
            'exact_value': optimum,
            'old_evaluations': original['evaluations'],
            'new_evaluations': plan.evaluations,
            'old_width': original['upper'] - original['lower'],
            'new_width': plan.upper - plan.lower,
            'diagnostic_seconds': elapsed,
        })
        policies.append({
            'task_id': original['task_id'],
            'policy': policy_to_data(plan.policy),
        })
    write_csv(out / 'tolerance_diagnostic.csv', results)
    (out / 'tolerance_policies.jsonl').write_text(
        ''.join(json.dumps(r, sort_keys=True) + '\n' for r in policies)
    )
    print(json.dumps({
        'tasks': len(results),
        'attain_exact': sum(abs(r['exact_value'] - r['new_value']) <= 1e-10 for r in results),
        'old_all_tolerance': all(r['old_status'] == 'tolerance' for r in results),
        'new_statuses': sorted({r['new_status'] for r in results}),
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('results/search'))
    args = parser.parse_args()
    main(args.source, args.out)
