"""Attribute search gains and analyze exact retest/decision mechanisms."""
import argparse
from collections import defaultdict, deque
from dataclasses import replace
from fractions import Fraction as F
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.analysis import _rows
from rl_retest.model import ModelSpec, PlanningProblem, EvaluationContext, best_stop, update, outcome_probability
from rl_retest.evaluation import evaluate
from rl_retest.planners.dynamic import solve_exact
from rl_retest.experiments.records import read_json, read_jsonl, dump_json, policy_from_data, policy_to_data
from scripts.analyze_study import write_csv, estimate

TOL=1e-10


def problem_from_data(raw):
    spec=dict(raw['spec']);spec['current']=tuple(spec['current']);spec['historical']=tuple(map(tuple,spec['historical']))
    return PlanningProblem(tuple(map(tuple,raw['belief'])),ModelSpec(**spec))


def threshold(out):
    rows=[]
    for n in (0,1,7):
        q=F(9,10)*F(1,2)**n/(F(1,10)+F(9,10)*F(1,2)**n)
        mu=q*F(4,5)
        # Integrate success and failure without division at zero-probability outcomes.
        g=mu*F(4,5)+max((1-mu)*F(1,2),mu*F(1,5))-max(F(1,2),mu)
        value=max(F(1,2),mu)+max(0,g-F(1,50))
        problem=PlanningProblem(((float(1-q),float(q),0.),),ModelSpec(1,.02,1,.5,(0,.8,0),()))
        plan=solve_exact(problem)
        ev=evaluate(plan.policy,EvaluationContext(problem.belief,problem.spec))
        assert abs(ev.expected_utility-float(value))<TOL
        rows.append(dict(failures=n,posterior_C=float(q),posterior_fraction=str(q),prediction=float(mu),
            gross_information=float(g),test_net_advantage=float(g-F(1,50)),
            stop_value=float(max(F(1,2),mu)),value=float(value),value_fraction=str(value),
            action=policy_to_data(plan.policy)['action'],terminal_path=getattr(plan.policy,'path'),
            policy=policy_to_data(plan.policy)))
    dump_json(out/'threshold_cases.json',rows)
    write_csv(out/'threshold_cases.csv',[{k:v for k,v in r.items() if k!='policy'} for r in rows])


def attribution(source,out):
    rows=[]
    for f in sorted((source/'planning/records').glob('*.json')):
        records=read_json(f);myopic=next(r for r in records if r['method']=='myopic')
        for warm in (r for r in records if r['method']=='warm'):
            cold=next(r for r in records if r['method']=='cold' and r['budget']==warm['budget'] and r['epsilon']==warm['epsilon'])
            s=warm['initialization']
            rows.append(dict(task_id=warm['task_id'],**warm['parameters'],budget=warm['budget'],epsilon=warm['epsilon'],
                initializer=s['status'],L0=s['lower'],U0=s['upper'],LT=warm['lower'],UT=warm['upper'],
                delta_policy=warm['lower']-s['lower'],delta_upper=s['upper']-warm['upper'],
                delta_width=s['upper']-s['lower']-warm['upper']+warm['lower'],
                final_width=warm['upper']-warm['lower'],same_policy=warm['policy']==myopic['policy'],
                same_value=abs(warm['value']-myopic['value'])<=TOL,
                value_minus_myopic=warm['value']-myopic['value'],lower_over_cold=warm['lower']-cold['lower'],
                upper_minus_cold=warm['upper']-cold['upper'],
                init_updates=s['evaluations'],search_updates=warm['evaluations']-s['evaluations'],
                total_updates=warm['evaluations'],reached_001=warm['upper']-warm['lower']<=.01,
                status=warm['status']))
    assert len(rows)==960
    write_csv(out/'search_attribution.csv',rows)
    summary=[]
    for n,b,e in sorted({(r['n'],r['budget'],r['epsilon']) for r in rows}):
        subset=[r for r in rows if (r['n'],r['budget'],r['epsilon'])==(n,b,e)]
        for metric in ('delta_policy','delta_upper','delta_width','lower_over_cold','final_width','same_policy','same_value','reached_001','init_updates','search_updates'):
            cells=defaultdict(list)
            for r in subset:cells[(r['k'],r['horizon'])].append(r[metric])
            summary.append(dict(n=n,budget=b,epsilon=e,metric=metric,**estimate(cells,seed=2026111702)))
    write_csv(out/'search_attribution_summary.csv',summary)


def difficulty(source,out):
    _,records=_rows(source/'main')
    pairs=defaultdict(dict)
    for r in records:
        if r['memory']=='correct' and r['method'] in ('exact','myopic'):
            pairs[r['task_id']][r['method']]=r
    assert len(pairs)==3240 and all(len(x)==2 for x in pairs.values())
    rows=[]
    for tid,pair in pairs.items():
        e,m=pair['exact'],pair['myopic']
        h=e['expected_utility']-e['initial_stop'];d=e['expected_utility']-m['expected_utility']
        assert h>=-TOL and d>=-TOL and d<=h+TOL
        kind='no_net_testing_value' if h<=TOL else 'testing_myopic_sufficient' if d<=TOL else 'sequential_gain'
        rows.append(dict(task_id=tid,**e['parameters'],H=h,D=d,category=kind,above_001=d>.001))
    write_csv(out/'difficulty_tasks.csv',rows)
    summaries=[]
    for kind in ('no_net_testing_value','testing_myopic_sufficient','sequential_gain'):
        rr=[r for r in rows if r['category']==kind]
        summaries.append(dict(category=kind,tasks=len(rr),fraction=len(rr)/len(rows),mean_H=np.mean([r['H'] for r in rr]),
            mean_D=np.mean([r['D'] for r in rr]),overall_D_contribution=sum(r['D'] for r in rr)/len(rows),above_001=sum(r['above_001'] for r in rr)))
    write_csv(out/'difficulty_classes.csv',summaries)
    estimates=[]
    for metric in ('H','D','above_001', 'testing_frequency', 'sequential_frequency'):
        cells=defaultdict(list)
        for r in rows:
            val=r['H']>TOL if metric=='testing_frequency' else r['D']>TOL if metric=='sequential_frequency' else r[metric]
            cells[tuple(r[k] for k in ('k','horizon','n','p_c_current','cost'))].append(val)
        estimates.append(dict(metric=metric,**estimate(cells)))
    write_csv(out/'difficulty_effects.csv',estimates)
    fig,ax=plt.subplots(figsize=(8,3))
    ax.bar(['No net test value','Testing; myopic sufficient','Sequential gain'],[r['fraction'] for r in summaries])
    ax.set_ylabel('Fraction of 3240 original tasks');fig.tight_layout();fig.savefig(out/'difficulty.png',dpi=150);plt.close(fig)


def first_divergence(a,b,problem):
    """First value-relevant different action along a shared reachable prefix."""
    queue=deque([(a,b,problem.belief,problem.spec.horizon,[],1.)])
    while queue:
        x,y,belief,h,history,reach=queue.popleft()
        ctx=EvaluationContext(belief,replace(problem.spec,horizon=h))
        xv=evaluate(policy_from_data(x),ctx);yv=evaluate(policy_from_data(y),ctx)
        if (x['action'],x['path'])!=(y['action'],y['path']):
            if abs(xv.expected_utility-yv.expected_utility)>TOL:
                branches=[]
                if x['action']=='test':
                    for o,key in ((0,'failure'),(1,'success')):
                        p=outcome_probability(belief,x['path'],o,problem.spec.current)
                        if p>0:
                            b2=update(belief,x['path'],o,problem.spec.current)
                            ev=evaluate(policy_from_data(x[key]),EvaluationContext(b2,replace(problem.spec,horizon=h-1)))
                            branches.append(dict(outcome=o,probability=p,posterior=b2,next_action=x[key]['action'],
                                                 path=x[key]['path'],continuation_value=ev.expected_utility,expected_tests=ev.expected_tests))
                return dict(history=history,reach_probability=reach,belief=belief,remaining=h,
                    exact_action=[x['action'],x['path']],other_action=[y['action'],y['path']],
                    exact_value=xv.expected_utility,other_value=yv.expected_utility,
                    local_gain=xv.expected_utility-yv.expected_utility,exact_branches=branches)
            continue
        if x['action']=='test':
            for o,key in ((0,'failure'),(1,'success')):
                p=outcome_probability(belief,x['path'],o,problem.spec.current)
                if p>0:
                    queue.append((x[key],y[key],update(belief,x['path'],o,problem.spec.current),h-1,history+[[x['path'],o]],reach*p))
    return None


def mechanisms(source,out):
    root=source/'research';jobs=read_jsonl(root/'jobs.jsonl')
    grid=[];deep=[];cases=[]
    for job in jobs:
        if job['group'] not in ('grid','deep'):continue
        data=read_json(root/'records'/f'{job["id"]}.json');p=problem_from_data(data['problem'])
        e,m=data['records'][:2];stop=best_stop(p.belief,p.spec)[0]
        if job['group']=='grid':
            grid.append(dict(**job,H=e['expected_utility']-stop,D=e['expected_utility']-m['expected_utility'],
                             exact_value=e['expected_utility'],myopic_value=m['expected_utility'],
                             action=e['policy']['action'],path=e['policy']['path'],
                             myopic_action=m['policy']['action']))
        else:
            for r in data['records']:
                deep.append(dict(id=job['id'],name=job['name'],k=job['k'],horizon=job['horizon'],cost=job['cost'],method=r['method'],
                    value=r['expected_utility'],regret=e['expected_utility']-r['expected_utility'],
                    H=e['expected_utility']-stop,width=(r['upper']-r['lower']) if r['upper'] is not None else '',
                    upper_slack=r['upper']-e['expected_utility'] if r['upper'] is not None else '',
                    evaluations=r['evaluations'],status=r['status'],policy_equal=r['policy']==e['policy']))
                if r['method']!='exact':
                    cases.append(dict(id=job['id'],method=r['method'],divergence=first_divergence(e['policy'],r['policy'],p)))
    assert len(grid)==1329 and len(deep)==120
    write_csv(out/'grid.csv',grid);write_csv(out/'deep.csv',deep);dump_json(out/'deep_divergences.json',cases)
    grid_summary=[]
    for h in (1,2,3):
        rr=[r for r in grid if r['horizon']==h]
        grid_summary.append(dict(horizon=h,points=len(rr),test_optimal=sum(r['H']>TOL for r in rr),
            myopic_gap_positive=sum(r['D']>TOL for r in rr),max_gap=max(r['D'] for r in rr)))
    write_csv(out/'grid_summary.csv',grid_summary)
    # Only the regular grid is rasterized; anchor records remain in full table.
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for col,h in enumerate((1,2,3)):
        rr=[r for r in grid if r['horizon']==h and int(r['id'].split('-')[1])<441]
        d=np.array([r['D'] for r in rr]).reshape(21,21)
        action=np.array([1 if r['action']=='test' else 2 if r['path']==0 else 0 for r in rr]).reshape(21,21)
        for row,values in enumerate((action,d)):
            im=axes[row,col].pcolormesh(np.linspace(0,1,21),np.linspace(0,.15,21),values.T,
                shading='nearest',vmin=0,vmax=2 if row==0 else max(r['D'] for r in grid))
            axes[row,col].set_xlim(0,1);axes[row,col].set_ylim(0,.15)
            axes[row,col].set_xlabel('Prior P(C)');axes[row,col].set_ylabel('Test cost')
            fig.colorbar(im,ax=axes[row,col],ticks=(0,1,2) if row==0 else None)
        axes[0,col].set_title(f'B={h}: 0 fallback, 1 test, 2 execute');axes[1,col].set_title('Exact minus rolling myopic')
    fig.tight_layout();fig.savefig(out/'mechanism_grid.png',dpi=150);plt.close(fig)
    summary=[]
    for method in ('myopic','blinkered','cold','warm'):
        rr=[r for r in deep if r['method']==method]
        summary.append(dict(method=method,tasks=len(rr),positive_regret=sum(r['regret']>TOL for r in rr),
            mean_regret=np.mean([r['regret'] for r in rr]),max_regret=max(r['regret'] for r in rr),
            mean_evaluations=np.mean([r['evaluations'] for r in rr]),
            optimal_but_uncertified=sum(r['regret']<=TOL and r['width']!='' and r['width']>TOL for r in rr)))
    write_csv(out/'deep_summary.csv',summary)


def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    threshold(out);attribution(source,out);difficulty(source,out);mechanisms(source,out)
    dump_json(out/'mechanism_provenance.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'research_manifest':read_json(source/'research/manifest.json'),'old_planning_manifest':read_json(source/'planning/manifest.json')})
    print('A/B/C analysis complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data'));p.add_argument('--out',type=Path,default=Path('results/research'))
    a=p.parse_args();main(a.source,a.out)
