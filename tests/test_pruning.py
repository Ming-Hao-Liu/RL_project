"""Check pairing, interrupted timing and task-level pruning statistics."""
import pytest

from scripts.pruning import pending_pairs, schedule, summarize


def test_timing_schedule_is_balanced_and_reproducible():
    task_ids = [f'task-{i:02d}' for i in range(20)]
    plan = schedule(task_ids)
    assert plan == schedule(task_ids)
    assert len(plan) == 60
    assert len({(p['task_id'], p['repeat']) for p in plan}) == 60
    for repeat in range(3):
        pairs = [p for p in plan if p['repeat'] == repeat]
        assert {p['task_id'] for p in pairs} == set(task_ids)
        assert all(sorted(p['order']) == [False, True] for p in pairs)
        assert sum(p['order'][0] for p in pairs) == 10


def test_resume_keeps_frozen_order_and_skips_only_complete_pairs():
    plan = schedule(['a', 'b'])
    pair = plan[2]
    records = [dict(task_id=pair['task_id'], repeat=pair['repeat'], prune=prune)
               for prune in pair['order']]
    assert pending_pairs(plan, []) == plan
    assert pending_pairs(plan, records) == [p for p in plan if p != pair]
    all_records = [dict(task_id=p['task_id'], repeat=p['repeat'], prune=prune)
                   for p in plan for prune in p['order']]
    assert pending_pairs(plan, all_records) == []


@pytest.mark.parametrize('kind', ['partial', 'duplicate', 'unknown_task', 'unknown_repeat'])
def test_resume_rejects_incomplete_or_unexpected_measurements(kind):
    plan = schedule(['a', 'b'])
    pair = plan[0]
    records = [dict(task_id=pair['task_id'], repeat=pair['repeat'], prune=prune)
               for prune in pair['order']]
    if kind == 'partial':
        records.pop()
    elif kind == 'duplicate':
        records.append(dict(records[0]))
    elif kind == 'unknown_task':
        for row in records:
            row['task_id'] = 'not-in-protocol'
    else:
        for row in records:
            row['repeat'] = 999
    with pytest.raises(ValueError):
        pending_pairs(plan, records)


def test_constant_ratio_has_exact_bootstrap_interval_and_absolute_effect():
    paired = [
        dict(task_id='a', on_wall=0.4, off_wall=0.2, on_cpu=0.3, off_cpu=0.15),
        dict(task_id='b', on_wall=0.8, off_wall=0.4, on_cpu=0.6, off_cpu=0.3),
    ]
    rows = summarize(paired)
    metrics = {row['metric']: row for row in rows}
    ratio = metrics['wall_ratio']
    assert ratio['estimate'] == pytest.approx(2)
    assert ratio['ci_low'] == pytest.approx(2)
    assert ratio['ci_high'] == pytest.approx(2)
    assert ratio['task_count'] == 2
    assert metrics['mean_wall_delta_ms']['estimate'] == pytest.approx(300)
    assert rows == summarize(paired)


def test_ratio_is_geometric_mean_with_tasks_as_sampling_units():
    paired = [
        dict(task_id='a', on_wall=0.9, off_wall=0.1, on_cpu=0.9, off_cpu=0.1),
        dict(task_id='b', on_wall=0.1, off_wall=0.1, on_cpu=0.1, off_cpu=0.1),
    ]
    ratio = next(row for row in summarize(paired) if row['metric'] == 'wall_ratio')
    assert ratio['estimate'] == pytest.approx(3)
    assert ratio['task_count'] == 2
    assert ratio['ci_low'] <= ratio['estimate'] <= ratio['ci_high']
