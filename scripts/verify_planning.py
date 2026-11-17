"""Re-evaluate saved confirmation policies without re-planning."""
import argparse
import hashlib
import json
from pathlib import Path
from rl_retest.experiments.tasks import Task, contexts
from rl_retest.experiments.records import policy_from_data
from rl_retest.evaluation import evaluate


def main(source, out):
    manifest=json.loads((source/'manifest.json').read_text())
    tasks={r['task_id']:Task.from_data(r) for r in
           map(json.loads,(source/'tasks.jsonl').read_text().splitlines())}
    files=list((source/'records').glob('*.json'))
    assert len(tasks)==len(files)==160
    checked=0
    for file in files:
        rows=json.loads(file.read_text())
        assert len(rows)==14
        keys={(r['method'],r['budget'],r['epsilon']) for r in rows}
        expected={(m,None,None) for m in ('myopic','blinkered')} | {
            (m,b,e) for m in ('cold','warm') for b in (100,1000,10000) for e in (0,.01)}
        assert keys==expected
        task=tasks[file.stem];_,truth=contexts(task,manifest['config'])
        for r in rows:
            assert r['task_id']==task.task_id
            v=evaluate(policy_from_data(r['policy']),truth).expected_utility
            assert abs(v-r['value'])<1e-10
            if r['budget'] is not None:
                assert r['evaluations']<=r['budget']
                assert abs(v-r['lower'])<1e-10 and v<=r['upper']+1e-10
            checked+=1
    timings=[json.loads(r) for r in (source/'timing.jsonl').read_text().splitlines()]
    assert len(timings)==384 and len({r['task_id'] for r in timings})==32
    assert len({(r['task_id'],r['method'],r['repeat']) for r in timings})==384
    for r in timings:
        saved=json.loads((source/'records'/f"{r['task_id']}.json").read_text())
        base=next(x for x in saved if x['method']==r['method'] and
                  (x['budget'] is None or (x['budget']==10000 and x['epsilon']==0)))
        assert abs(r['value']-base['value'])<1e-10
        assert r['evaluations']==base['evaluations']
    out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps({'policies':checked,'tasks':160,'timing_records':384,'valid':True,'scope':'Fixed policy values, budget and completeness; no exact optimum for large tasks'},indent=2))
    (out/'input_sha256.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(source.rglob('*')) if f.is_file()},indent=2))
    print(f'{checked} saved policies and 384 timing records verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/planning'))
    p.add_argument('--out',type=Path,default=Path('results/planning'))
    a=p.parse_args();main(a.source,a.out)
