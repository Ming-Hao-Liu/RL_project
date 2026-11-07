from pathlib import Path
import pytest
from rl_retest.experiments.tasks import ExperimentSpec, create_tasks
from rl_retest.experiments.records import policy_to_data, policy_from_data, read_json
from rl_retest.experiments.runner import prepare, run, verify
from rl_retest.policy import Stop, Test as PolicyTest


def test_policy_roundtrip():
    policy = PolicyTest(1, Stop(-1), PolicyTest(0, Stop(1), Stop(0)))
    assert policy_from_data(policy_to_data(policy)) == policy


def test_config_sizes_and_pairing():
    config = read_json('configs/study.json')
    main = create_tasks(ExperimentSpec(config,'main'))
    large = create_tasks(ExperimentSpec(config,'large'))
    stress = create_tasks(ExperimentSpec(config,'stress'))
    assert len(main) == 3240 and len(large) == 160 and len(stress) == 40
    assert {t.task_id for t in stress} <= {t.task_id for t in main}
    assert all(next(m for m in main if m.task_id == t.task_id) == t for t in stress)
    grouped = {}
    for t in large:
        p = t.parameters
        key = (p['k'],p['horizon'],p['index'])
        grouped.setdefault(key,[]).append(t)
    assert all(len(ts)==2 and ts[0].priors == ts[1].priors for ts in grouped.values())


def test_prepare_run_resume_verify(tmp_path):
    root = tmp_path / 'run'
    manifest = prepare('configs/study.json','dev',root)
    assert manifest['task_count'] == 16
    assert run(root,1) == manifest['job_count']
    assert verify(root)['valid']
    assert run(root,1) == 0
    with pytest.raises(FileExistsError):
        prepare('configs/study.json','dev',root)


def test_serial_parallel_and_analysis(tmp_path):
    from rl_retest.analysis import analyze
    from rl_retest.experiments.records import read_jsonl
    import os
    # Keep plotting caches in the test directory rather than a user cache.
    os.environ['MPLCONFIGDIR'] = str(tmp_path / 'mpl')
    roots = [tmp_path / 'serial', tmp_path / 'parallel']
    for root, workers in zip(roots, (1,2)):
        manifest = prepare('configs/study.json','dev',root)
        run(root,workers)
        assert verify(root)['valid']
    def records(root):
        result = {}
        for path in (root / 'records').glob('*.jsonl'):
            for row in read_jsonl(path):
                for key in ('planning_seconds','evaluation_seconds','timing_contended'):
                    row.pop(key, None)
                result[(row['job_id'],row['method'],row['memory'],row.get('budget'),row.get('epsilon'))] = row
        return result
    assert records(roots[0]) == records(roots[1])
    result = analyze(roots[0])
    assert result['completed_jobs'] == manifest['job_count']
    assert (roots[0] / 'analysis/tables/summary.csv').exists()
    assert (roots[0] / 'analysis/figures/paired.png').exists()


def test_incompatible_planning_history_uses_fallback():
    from rl_retest.experiments.tasks import Task
    from rl_retest.experiments.runner import _execute
    from rl_retest.model import History
    config = read_json('configs/study.json')
    config['model']['p_c_old'] = 1.0
    task = Task('impossible', {'k':1,'horizon':1,'n':1,'p_c_current':0.0,'cost':.02,'index':0},
                ((0.,1.,0.),), History(((1,),),((0,),)), 1)
    job = {'job_id':'one','task_id':'impossible','method':'exact','memory':'misread'}
    record = _execute((task.to_data(),config,job,False))[0]
    assert record['status'] == 'incompatible_history'
    assert record['policy'] == {'action':'stop','path':-1}
    assert record['model_value'] is None and record['upper'] is None
    assert record['expected_utility'] == .5


def test_run_rejects_changed_source_or_config(tmp_path, monkeypatch):
    from rl_retest.experiments import runner
    root = tmp_path / 'versioned'
    runner.prepare('configs/study.json','dev',root)
    monkeypatch.setattr(runner,'_source_digest',lambda: 'different')
    with pytest.raises(ValueError, match='source has changed'):
        runner.run(root,1)


def test_aliases_and_checkpoints_are_individually_verified(tmp_path):
    import json
    from rl_retest.experiments.records import read_jsonl
    root = tmp_path / 'large'
    config = read_json('configs/study.json')
    config['suites']['large'] = {'k':[2], 'horizon':[2], 'history':[0],
                                 'p_c_current':[.5], 'cost':[.02], 'tasks':1}
    config['search']['budgets'] = [1,5]
    config['search']['epsilons'] = [.01,.05]
    config_file = tmp_path / 'small.json'
    config_file.write_text(json.dumps(config))
    manifest = prepare(config_file,'large',root)
    run(root,1)
    assert verify(root)['valid']
    jobs = read_jsonl(root / 'jobs.jsonl')
    trace = next(j for j in jobs if j['method']=='bounded_trace')
    path = root / 'records' / (trace['job_id'] + '.jsonl')
    records = read_jsonl(path)
    assert len(records) == 4
    path.write_text('\n'.join(json.dumps(r) for r in records[:-1]) + '\n')
    assert not verify(root)['valid']
