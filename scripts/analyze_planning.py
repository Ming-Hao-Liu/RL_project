"""Paired confirmation, initialization decomposition and matched timing."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analyze_study import estimate, write_csv


def main(source, out):
    manifest=json.loads((source/'manifest.json').read_text())
    rows=[r for f in sorted((source/'records').glob('*.json')) for r in json.loads(f.read_text())]
    by_task=defaultdict(dict)
    for row in rows:
        key=(row['method'],row['budget'],row['epsilon'])
        assert key not in by_task[row['task_id']]
        by_task[row['task_id']][key]=row
    assert len(by_task)==160 and all(len(r)==14 for r in by_task.values())
    out.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for tid,records in by_task.items():
        baseline=records[('myopic',None,None)]
        p=baseline['parameters']
        for budget in (100,1000,10000):
            for epsilon in (0,.01):
                cold,warm=[records[(m,budget,epsilon)] for m in ('cold','warm')]
                seed=warm['initialization']
                pairs.append(dict(task_id=tid,**p,budget=budget,epsilon=epsilon,
                    width_delta=(warm['upper']-warm['lower'])-(cold['upper']-cold['lower']),
                    utility_delta=warm['value']-cold['value'],above_myopic=warm['value']-baseline['value'],
                    upper_delta=warm['upper']-cold['upper'],lower_delta=warm['lower']-cold['lower'],
                    search_lower_gain=warm['lower']-seed['lower'],search_upper_reduction=seed['upper']-warm['upper'],
                    seed_lower=seed['lower'],seed_upper=seed['upper'],seed_evaluations=seed['evaluations'],
                    initialization_complete=seed['status']=='complete',initialization_status=seed['status'],
                    warm_width=warm['upper']-warm['lower'],cold_width=cold['upper']-cold['lower'],
                    warm_value=warm['value'],cold_value=cold['value'],myopic_value=baseline['value'],
                    warm_evaluations=warm['evaluations'],cold_evaluations=cold['evaluations'],
                    warm_status=warm['status'],cold_status=cold['status']))
    write_csv(out/'paired_tasks.csv',pairs)
    summary=[]
    metrics=('width_delta','utility_delta','above_myopic','upper_delta','lower_delta','search_lower_gain','search_upper_reduction')
    for n in (0,3):
        for budget in (100,1000,10000):
            for epsilon in (0,.01):
                subset=[r for r in pairs if (r['n'],r['budget'],r['epsilon'])==(n,budget,epsilon)]
                for metric in metrics:
                    cells=defaultdict(list)
                    for r in subset:cells[(r['k'],r['horizon'])].append(r[metric])
                    summary.append(dict(n=n,budget=budget,epsilon=epsilon,metric=metric,
                        primary=n==0 and budget==10000 and epsilon==0 and metric=='width_delta',
                        **estimate(cells,seed=manifest['protocol']['bootstrap_seed'])))
    write_csv(out/'effects.csv',summary)
    costs=[]
    for n in (0,3):
        for method in ('cold','warm','myopic','blinkered'):
            for budget,epsilon in ([(b,e) for b in (100,1000,10000) for e in (0,.01)] if method in ('cold','warm') else [(None,None)]):
                rr=[r for r in rows if (r['parameters']['n'],r['method'],r['budget'],r['epsilon'])==(n,method,budget,epsilon)]
                costs.append(dict(n=n,method=method,budget=budget,epsilon=epsilon,tasks=len(rr),
                    value=np.mean([r['value'] for r in rr]),updates=np.mean([r['evaluations'] for r in rr]),
                    width=np.mean([r['upper']-r['lower'] for r in rr]) if method in ('cold','warm') else '',
                    achieved_fraction=np.mean([r['status'] in ('complete','tolerance') for r in rr]) if method in ('cold','warm') else '',
                    initialization_updates=np.mean([r['initialization']['evaluations'] for r in rr]) if method=='warm' else '',
                    statuses=json.dumps({s:sum(r['status']==s for r in rr) for s in sorted({r['status'] for r in rr})})))
    write_csv(out/'costs.csv',costs)
    timings=[json.loads(line) for line in (source/'timing.jsonl').read_text().splitlines()]
    assert len(timings)==384
    timing=[]
    for n in (0,3):
        for method in ('cold','warm','myopic','blinkered'):
            subset=[r for r in timings if r['parameters']['n']==n and r['method']==method]
            task_ids=sorted({r['task_id'] for r in subset})
            assert len(task_ids)==16
            item=dict(n=n,method=method,tasks=16,repeats=3)
            for field in ('wall','cpu'):
                values=[np.median([r[field] for r in subset if r['task_id']==tid]) for tid in task_ids]
                item[f'{field}_median']=np.median(values);item[f'{field}_p90']=np.quantile(values,.9)
            timing.append(item)
    write_csv(out/'timing.csv',timing)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for method in ('cold','warm'):
        rr=[r for r in costs if r['n']==0 and r['method']==method and r['epsilon']==0]
        axes[0].plot([r['updates'] for r in rr],[r['value'] for r in rr],'-o',label=method)
        axes[1].plot([r['updates'] for r in rr],[r['width'] for r in rr],'-o',label=method)
    myopic=next(r for r in costs if r['n']==0 and r['method']=='myopic')
    axes[0].axhline(myopic['value'],ls=':',color='black',label='full myopic')
    for ax,ylabel in zip(axes[:2],('Expected utility (R=1)','Certificate width')):
        ax.set_xlabel('Total successor updates');ax.set_ylabel(ylabel);ax.legend()
    rr=[r for r in timing if r['n']==0]
    axes[2].bar([r['method'] for r in rr],[r['wall_median'] for r in rr])
    axes[2].set_ylabel('Planning wall time (seconds)');axes[2].set_title('M=10000, epsilon=0; matched tasks')
    fig.suptitle('Independent cohort: n=0, 80 tasks; timing: 16 tasks x 3 repeats')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'planning_comparison.{ext}',dpi=220)
    plt.close(fig)
    main_pairs=[r for r in pairs if r['n']==0 and r['budget']==10000 and r['epsilon']==0]
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(['after initialization','after search'],
                 [np.mean([r['seed_upper'] for r in main_pairs]),np.mean([r['seed_upper']-r['search_upper_reduction'] for r in main_pairs])],'-o',label='upper')
    axes[0].plot(['after initialization','after search'],
                 [np.mean([r['seed_lower'] for r in main_pairs]),np.mean([r['warm_value'] for r in main_pairs])],'-o',label='lower / policy value')
    axes[0].set_ylabel('Expected utility bound (R=1)');axes[0].legend()
    axes[1].bar(['lower increase','upper decrease'],
                [np.mean([r['lower_delta'] for r in main_pairs]),-np.mean([r['upper_delta'] for r in main_pairs])])
    axes[1].set_ylabel('Contribution to narrower width vs cold')
    axes[1].set_title('Same total M=10000, epsilon=0')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'initialization_decomposition.{ext}',dpi=220)
    plt.close(fig)
    primary=next(r for r in summary if r['primary'])
    (out/'README.md').write_text('# 根节点初始化确认结果\n\n输入为独立种子固定160任务；paired_tasks保留全体对照，effects给设置等权配对区间；仅标记primary的比较是预定主终点。costs保留全部结束状态，timing先在同任务内取三次中位数，再跨16任务汇总。无历史与有历史分开。完整策略和384条原始计时在data/planning中。\n\n主要比较（warm−cold证书宽度）：'+json.dumps(primary,ensure_ascii=False)+'\n')
    (out/'provenance.json').write_text(json.dumps({'manifest':manifest,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tasks':160,'records':len(rows),'timing_records':len(timings)},indent=2))
    print(json.dumps({'primary':primary,'timing':timing},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/planning'))
    p.add_argument('--out',type=Path,default=Path('results/planning'))
    args=p.parse_args();main(args.source,args.out)
