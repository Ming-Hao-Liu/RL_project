"""Clustered calibration uncertainty on a fixed task set, without replanning."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.experiments.records import read_json,read_jsonl,dump_json
from rl_retest.experiments.tasks import Task,contexts
from rl_retest.model import best_stop, update, EvaluationContext
from rl_retest.evaluation import evaluate
from rl_retest.experiments.records import policy_from_data
from dataclasses import replace
from scripts.analyze_study import write_csv
from scripts.analyze_mechanisms import first_divergence, problem_from_data

TOL=1e-10


def cluster_interval(values,seed=2026111702):
    """One scalar per independent calibration repetition; never flatten tasks."""
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    assert len(a)==30
    samples=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(mean=float(a.mean()),ci_low=float(lo),ci_high=float(hi),clusters=len(a))


def boundary_evidence(example):
    divergence=example['first_value_relevant_divergence']
    if divergence is None:
        return None
    model=problem_from_data({'belief':example['planning_belief'],'spec':example['planning_spec']})
    belief=model.belief
    a,b=example['true_policy'],example['estimated_policy']
    for path,outcome in divergence['history']:
        belief=update(belief,path,outcome,model.spec.current)
        key='success' if outcome else 'failure'
        a,b=a[key],b[key]
    spec=replace(model.spec,horizon=divergence['remaining'])
    ctx=EvaluationContext(belief,spec)
    chosen=evaluate(policy_from_data(b),ctx).expected_utility
    alternative=evaluate(policy_from_data(a),ctx).expected_utility
    return dict(estimated_belief=belief,estimated_stop=best_stop(belief,spec)[0],
                estimated_chosen_continuation=chosen,estimated_true_policy_continuation=alternative,
                estimated_preference=chosen-alternative,
                true_preference=divergence['exact_value']-divergence['other_value'])


def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    root=source/'research';manifest=read_json(root/'manifest.json')
    tasks={t['task_id']:t for t in read_jsonl(root/'tasks.jsonl')}
    true=read_json(root/'truth.json')
    jobs=[j for j in read_jsonl(root/'jobs.jsonl') if j['group']=='calibration']
    assert len(jobs)==120
    rows=[];clusters=[];raw_cal=[]
    examples={}
    config=read_json(source/'main/manifest.json')['config']
    true_contexts={tid:contexts(Task.from_data(raw),config) for tid,raw in tasks.items()}
    for job in jobs:
        data=read_json(root/'records'/f'{job["id"]}.json');cal=data['calibration'];raw_cal.append(cal)
        es=cal['estimates'];expected={'c_old':.1,'c_0.1':.1,'c_0.5':.5,'c_0.9':.9,'a':.9}
        errors=np.array([es[k]-p for k,p in expected.items()])
        clusters.append(dict(size=job['size'],repeat=job['repeat'],parameter_mae=float(np.abs(errors).mean()),
            parameter_rmse=float(np.sqrt(np.mean(errors**2))),**es))
        for r in data['records']:
            baseline=true[r['task_id']];task=tasks[r['task_id']];params=task['parameters']
            regret=baseline['expected_utility']-r['expected_utility']
            assert regret>=-TOL
            same=r['policy']==baseline['policy']
            true_belief=true_contexts[r['task_id']][1].belief
            belief_tv=float(np.mean([sum(abs(x-y) for x,y in zip(a,b))/2
                                     for a,b in zip(r['planning_belief'],true_belief)]))
            root_same=(r['policy']['action'],r['policy']['path'])==(baseline['policy']['action'],baseline['policy']['path'])
            kind='identical' if same else 'equal_value_different_policy' if abs(regret)<=TOL else 'root_change' if not root_same else 'later_change'
            rows.append(dict(size=job['size'],repeat=job['repeat'],task_id=r['task_id'],**params,
                component=r['component'],method=r['method'],regret=regret,
                mean_path_posterior_tv=belief_tv,
                model_error=r['model_value']-r['expected_utility'],absolute_model_error=abs(r['model_value']-r['expected_utility']),
                true_value=r['expected_utility'],optimal_value=baseline['expected_utility'],
                expected_tests=r['expected_tests'],test_delta=r['expected_tests']-baseline['expected_tests'],
                policy_changed=not same,root_changed=not root_same,loss_positive=regret>TOL,category=kind,
                old_c_error=es['c_old']-.1,current_c_error=es[f"c_{params['p_c_current']}"]-params['p_c_current'],a_error=es['a']-.9))
            # Deterministic first occurrence, labelled explanatory, never a population sample.
            key=(kind,r['component'])
            if r['method']=='exact' and job['size']==10 and key not in examples:
                problem,_=contexts(Task.from_data(task),config)
                examples[key]={'selection':'first in frozen job/task order; explanatory example',
                    'job':job,'task':task,'calibration':cal,'true_policy':baseline['policy'],
                    'estimated_policy':r['policy'],'regret':regret,'model_value':r['model_value'],
                    'true_value':r['expected_utility'],'planning_belief':r['planning_belief'],
                    'planning_spec':r['planning_spec'], 'category':kind,'component':r['component'],
                    'first_value_relevant_divergence':first_divergence(baseline['policy'],r['policy'],problem)}
    assert len(rows)==8880 and len(clusters)==120
    write_csv(out/'calibration_tasks.csv',rows);write_csv(out/'calibration_parameters.csv',clusters)
    for example in examples.values():
        example['decision_boundary_evidence']=boundary_evidence(example)
    dump_json(out/'calibration_examples.json',list(examples.values()))
    dump_json(out/'calibration_tables.json',raw_cal)
    cluster_rows=[]
    groups=[('all','all')]+[(k,v) for k in ('n','cost','p_c_current') for v in sorted({r[k] for r in rows})]
    for size in (10,30,100,300):
        for method in ('exact','myopic'):
            for field,label in groups:
                for rep in range(30):
                    rr=[r for r in rows if r['component']=='both' and r['size']==size and r['repeat']==rep and r['method']==method and (field=='all' or r[field]==label)]
                    assert rr
                    item=dict(size=size,method=method,group=field,label=label,repeat=rep,tasks=len(rr))
                    for metric in ('regret','absolute_model_error','model_error','expected_tests','test_delta','policy_changed','root_changed','loss_positive','mean_path_posterior_tv'):
                        item[metric]=float(np.mean([r[metric] for r in rr]))
                    cluster_rows.append(item)
    write_csv(out/'calibration_clusters.csv',cluster_rows)
    summaries=[]
    for size,method,field,label in sorted({(r['size'],r['method'],r['group'],str(r['label'])) for r in cluster_rows}):
        rr=[r for r in cluster_rows if (r['size'],r['method'],r['group'],str(r['label']))==(size,method,field,label)]
        assert len(rr)==30
        for metric in ('regret','absolute_model_error','model_error','expected_tests','test_delta','policy_changed','root_changed','loss_positive','mean_path_posterior_tv'):
            summaries.append(dict(size=size,method=method,group=field,label=label,metric=metric,
                                  **cluster_interval([r[metric] for r in rr])))
    write_csv(out/'calibration_effects.csv',summaries)
    primary=[]
    for method in ('exact','myopic'):
        by={(r['size'],r['repeat']):r for r in cluster_rows if r['method']==method and r['group']=='all'}
        diff=[by[(300,rep)]['regret']-by[(10,rep)]['regret'] for rep in range(30)]
        primary.append(dict(comparison='m300-minus-m10 regret',method=method,primary=method=='exact',**cluster_interval(diff)))
    for size in (10,30,100,300):
        by={(r['method'],r['repeat']):r for r in cluster_rows if r['size']==size and r['group']=='all'}
        diff=[by[('myopic',rep)]['regret']-by[('exact',rep)]['regret'] for rep in range(30)]
        primary.append(dict(comparison='estimated myopic minus estimated DP regret',size=size,primary=False,**cluster_interval(diff)))
    write_csv(out/'calibration_comparisons.csv',primary)
    parameter_summary=[]
    for size in (10,30,100,300):
        for metric in ('parameter_mae','parameter_rmse'):
            parameter_summary.append(dict(size=size,metric=metric,**cluster_interval([r[metric] for r in clusters if r['size']==size])))
    write_csv(out/'calibration_parameter_effects.csv',parameter_summary)
    diag=[]
    for size in (10,300):
        for component in ('both','history','future'):
            per_rep=[]
            for rep in range(30):
                rr=[r for r in rows if r['size']==size and r['repeat']==rep and r['method']=='exact' and r['component']==component and r['n']==3 and r['p_c_current']==.5 and r['cost']==.02]
                assert len(rr)==2
                per_rep.append(np.mean([r['regret'] for r in rr]))
            diag.append(dict(size=size,component=component,**cluster_interval(per_rep)))
    write_csv(out/'calibration_error_sources.csv',diag)
    categories=[]
    for size in (10,30,100,300):
        for method in ('exact','myopic'):
            for category in ('identical','equal_value_different_policy','root_change','later_change'):
                rr=[r for r in rows if r['size']==size and r['method']==method and r['component']=='both' and r['category']==category]
                categories.append(dict(size=size,method=method,category=category,records=len(rr),fraction=len(rr)/1080,
                    conditional_regret=float(np.mean([r['regret'] for r in rr])) if rr else None))
    write_csv(out/'calibration_action_categories.csv',categories)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    rr=[r for r in parameter_summary if r['metric']=='parameter_mae']
    axes[0].errorbar([r['size'] for r in rr],[r['mean'] for r in rr],
                     yerr=[[r['mean']-r['ci_low'] for r in rr],[r['ci_high']-r['mean'] for r in rr]],fmt='o-')
    axes[0].set_ylabel('Mean absolute probability error')
    for method in ('exact','myopic'):
        for ax,metric in zip(axes[1:],('regret','policy_changed')):
            rr=[r for r in summaries if r['method']==method and r['group']=='all' and r['metric']==metric]
            ax.errorbar([r['size'] for r in rr],[r['mean'] for r in rr],
                yerr=[[r['mean']-r['ci_low'] for r in rr],[r['ci_high']-r['mean'] for r in rr]],fmt='o-',label='plug-in DP' if method=='exact' else 'plug-in myopic')
    axes[1].set_ylabel('True regret vs true-table DP');axes[2].set_ylabel('Fraction of policies changed')
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Calibration samples per C parameter')
    axes[1].legend();axes[2].legend();fig.tight_layout();fig.savefig(out/'calibration.png',dpi=150);plt.close(fig)
    dump_json(out/'calibration_provenance.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'clusters_per_size':30,'tasks':36,'raw_records':len(rows),'scope':'fixed tasks; CIs reflect independent calibration repeats only',
        'manifest':manifest})
    print(primary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('data'));p.add_argument('--out',type=Path,default=Path('results/research'))
    a=p.parse_args();main(a.source,a.out)
