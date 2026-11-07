"""Paired descriptive analysis of saved records; never invokes planning."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .experiments.records import dump_json, read_json, read_jsonl


def _rows(root):
    manifest = read_json(root / 'manifest.json')
    config_hash = hashlib.sha256(json.dumps(manifest['config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if config_hash != manifest.get('config_sha256'):
        raise ValueError('configuration snapshot does not match manifest')
    for name in ('jobs','tasks'):
        digest = hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest()
        if digest != manifest[f'{name}_sha256']:
            raise ValueError(f'{name} file does not match manifest')
    jobs = read_jsonl(root / 'jobs.jsonl')
    rows, errors = [], []
    for job in jobs:
        path = root / 'records' / f"{job['job_id']}.jsonl"
        if not path.exists():
            errors.append(job['job_id'])
            continue
        records = read_jsonl(path)
        expected_count = (len(job['budgets']) * len(job['epsilons']) if job['method'] == 'bounded_trace'
                          else len(job['budgets']) if job['method'] == 'iterative_trace'
                          else 1 + len(job.get('aliases', [])))
        if (len(records) != expected_count or any(record['status'] == 'failed' or
                record['job_id'] != job['job_id'] for record in records)):
            errors.append(job['job_id'])
        else:
            if job['method'] == 'bounded_trace':
                expected = {(b, e) for b in job['budgets'] for e in job['epsilons']}
                if {(r['budget'], r['epsilon']) for r in records} != expected:
                    errors.append(job['job_id'])
                    continue
            elif job['method'] == 'iterative_trace':
                if {r['budget'] for r in records} != set(job['budgets']):
                    errors.append(job['job_id'])
                    continue
            else:
                expected = {(job['method'], job['memory'])} | {(a['method'], a['memory']) for a in job.get('aliases', [])}
                if {(r['method'], r['memory']) for r in records} != expected:
                    errors.append(job['job_id'])
                    continue
            rows.extend(records)
    if errors:
        raise ValueError(f'{len(errors)} jobs missing, failed or mismatched; analysis requires complete coverage')
    return manifest, rows


def _csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell(row):
    p = row['parameters']
    return tuple(p[k] for k in ('k','horizon','n','p_c_current','cost'))


def _paired(rows, left, right, samples, seed):
    by_task = defaultdict(dict)
    for row in rows:
        key = (row['method'], row['memory'])
        if key in (left,right) and not row.get('scenario'):
            by_task[row['task_id']][key] = row
    cells = defaultdict(list)
    for values in by_task.values():
        if left in values and right in values:
            a,b = values[left],values[right]
            cells[_cell(a)].append(a['expected_utility']-b['expected_utility'])
    if not cells:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.zeros(samples)
    means = []
    for key in sorted(cells):
        values = np.array(cells[key])
        means.append(values.mean())
        estimates += values[rng.integers(0,len(values),size=(samples,len(values)))].mean(axis=1)
    estimates /= len(cells)
    lo,hi = np.quantile(estimates,[.025,.975])
    return {'left': '/'.join(left), 'right': '/'.join(right), 'mean_difference': float(np.mean(means)),
            'ci_low':float(lo), 'ci_high':float(hi), 'cells':len(cells), 'pairs':sum(map(len,cells.values()))}


def analyze(run_dir):
    root = Path(run_dir)
    manifest, rows = _rows(root)
    out = root / 'analysis'
    tables, figures = out / 'tables', out / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        key = (row['method'],row['memory'],row.get('scenario','nominal'),row.get('budget',''),
               row.get('epsilon',''),row.get('prune',''),row.get('assumed_current',''))
        groups[key].append(row)
    summary=[]
    for key, items in sorted(groups.items(),key=lambda x:str(x[0])):
        times = [r['planning_seconds'] for r in items if not r['timing_contended']]
        widths = [r['upper']-r['lower'] for r in items if r['upper'] is not None and r['lower'] is not None]
        bias = [r['model_value']-r['expected_utility'] for r in items if r['model_value'] is not None]
        summary.append(dict(zip(('method','memory','scenario','budget','epsilon','prune','assumed_current'),key)) | {
            'count':len(items), 'mean_utility':float(np.mean([r['expected_utility'] for r in items])),
            'mean_tests':float(np.mean([r['expected_tests'] for r in items])),
            'mean_width':float(np.mean(widths)) if widths else '',
            'mean_model_minus_true':float(np.mean(bias)) if bias else '',
            'aliases':sum('alias_of' in r for r in items),
            'median_seconds':float(np.median(times)) if times else '',
            'p90_seconds':float(np.quantile(times,.9)) if times else ''})
    _csv(tables / 'summary.csv',summary)
    refs = {r['task_id']:r['expected_utility'] for r in rows if r['method']=='exact' and r['memory']=='correct'
            and r.get('scenario','nominal') == 'nominal' and 'assumed_current' not in r}
    cell_groups = defaultdict(list)
    for row in rows:
        key = (_cell(row),row['method'],row['memory'],row.get('scenario','nominal'),
               row.get('budget',''),row.get('epsilon',''),row.get('prune',''),row.get('assumed_current',''))
        cell_groups[key].append(row)
    cell_summary=[]
    for key, items in sorted(cell_groups.items(),key=lambda x:str(x[0])):
        cell,method,memory,scenario,budget,epsilon,prune,assumed = key
        regrets = [refs[r['task_id']]-r['expected_utility'] for r in items
                   if r['task_id'] in refs and scenario != 'type_resampling']
        cell_summary.append(dict(zip(('k','horizon','history','p_c_current','cost'),cell)) | {
            'method':method,'memory':memory,'scenario':scenario,'budget':budget,'epsilon':epsilon,
            'prune':prune,'assumed_current':assumed,'count':len(items),
            'mean_utility':float(np.mean([r['expected_utility'] for r in items])),
            'mean_tests':float(np.mean([r['expected_tests'] for r in items])),
            'mean_regret':float(np.mean(regrets)) if regrets else '',
            'mean_evaluations':float(np.mean([r['evaluations'] for r in items]))})
    _csv(tables / 'by_setting.csv',cell_summary)
    comparisons=[]
    for a,b in [(('exact','correct'),('exact','reset')), (('exact','correct'),('exact','misread')),
                (('exact','correct'),('myopic','correct'))]:
        result = _paired(rows,a,b,manifest['config']['bootstrap_samples'],manifest['config']['seeds']['bootstrap'])
        if result:
            comparisons.append(result)
    _csv(tables / 'paired.csv',comparisons)
    # Import only when plotting, avoiding import-time configuration and cache writes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if comparisons:
        fig,ax = plt.subplots(figsize=(8,3.8))
        for i, item in enumerate(comparisons):
            mean = item['mean_difference']
            ax.plot([item['ci_low'],item['ci_high']],[i,i],color='C0')
            ax.plot(mean,i,'o',color='C0')
        ax.axvline(0,color='black',linewidth=.7)
        ax.set_yticks(range(len(comparisons)),[f"{x['left']} − {x['right']}" for x in comparisons])
        ax.set_xlabel('Paired expected-utility difference (95% bootstrap interval)')
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(figures / f'paired.{ext}', bbox_inches='tight')
        plt.close(fig)
    if manifest['suite'] in ('main','dev') and comparisons:
        costs = sorted({r['parameters']['cost'] for r in rows})
        probabilities = sorted({r['parameters']['p_c_current'] for r in rows})
        fig,axes = plt.subplots(1,len(costs),figsize=(4*len(costs),3.5),squeeze=False)
        for ax,cost in zip(axes[0],costs):
            for comparison, color in [('reset','C0'),('misread','C1')]:
                points=[]
                for probability in probabilities:
                    subset=[r for r in rows if r['parameters']['cost']==cost and r['parameters']['p_c_current']==probability]
                    estimate=_paired(subset,('exact','correct'),('exact',comparison),
                                     manifest['config']['bootstrap_samples'],manifest['config']['seeds']['bootstrap'])
                    if estimate: points.append((probability,estimate))
                if points:
                    xs=[x for x,_ in points]; ys=[p['mean_difference'] for _,p in points]
                    ax.plot(xs,ys,marker='o',label=comparison,color=color)
                    ax.fill_between(xs,[p['ci_low'] for _,p in points],[p['ci_high'] for _,p in points],color=color,alpha=.15)
            ax.axhline(0,color='black',linewidth=.6); ax.set_title(f'Cost = {cost}')
            ax.set_xlabel('Current capability-dependent success probability')
        axes[0][0].set_ylabel('Correct-history utility advantage')
        axes[0][-1].legend()
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(figures / f'memory_effect.{ext}', bbox_inches='tight')
        plt.close(fig)
    if manifest['suite'] in ('large','pruning'):
        selected=[r for r in rows if r['method']=='bounded' and r.get('epsilon') == .01]
        if selected:
            fig,ax=plt.subplots(figsize=(7,4))
            for n in sorted({r['parameters']['n'] for r in selected}):
                series=[]
                for budget in sorted({r['budget'] for r in selected}):
                    subset=[r for r in selected if r['parameters']['n']==n and r['budget']==budget]
                    if subset: series.append((budget,float(np.mean([r['upper']-r['lower'] for r in subset]))))
                if series: ax.plot([x[0] for x in series],[x[1] for x in series],marker='o',label=f'n={n}')
            ax.set_xscale('log'); ax.set_xlabel('Successor-belief evaluation budget'); ax.set_ylabel('Mean model bound width'); ax.legend()
            fig.tight_layout()
            for ext in ('png','pdf'): fig.savefig(figures / f'bound_width.{ext}', bbox_inches='tight')
            plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4))
            for n in sorted({r['parameters']['n'] for r in selected}):
                for method in ('bounded','iterative'):
                    series=[]
                    for budget in sorted({r['budget'] for r in selected}):
                        subset=[r for r in rows if r['method']==method and r['parameters']['n']==n
                                and r.get('budget')==budget and (method!='bounded' or r.get('epsilon')==.01)]
                        if subset: series.append((budget,float(np.mean([r['expected_utility'] for r in subset]))))
                    if series: ax.plot([x[0] for x in series],[x[1] for x in series],marker='o',label=f'{method}, n={n}')
            ax.set_xscale('log'); ax.set_xlabel('Successor-belief evaluation budget')
            ax.set_ylabel('Mean expected net utility'); ax.legend()
            fig.tight_layout()
            for ext in ('png','pdf'): fig.savefig(figures / f'utility_vs_budget.{ext}', bbox_inches='tight')
            plt.close(fig)
            largest=max(r['budget'] for r in selected)
            initial=[r for r in selected if r['budget']==largest]
            fig,ax=plt.subplots(figsize=(6,4))
            for pos,n in enumerate(sorted({r['parameters']['n'] for r in initial})):
                subset=[r for r in initial if r['parameters']['n']==n]
                certified=sum(r['perfect_information']-r['initial_stop'] <= r['parameters']['cost'] for r in subset)
                tolerance=sum(r['perfect_information']-r['initial_stop'] > r['parameters']['cost'] and
                              max(0,r['perfect_information']-r['parameters']['cost']-r['initial_stop']) <=
                              r['epsilon'] * manifest['config']['model']['reward']
                              for r in subset)
                ax.bar(pos,certified/len(subset),color='C0',label='Stop provably optimal' if pos==0 else None)
                ax.bar(pos,tolerance/len(subset),bottom=certified/len(subset),color='C1',
                       label='Initial tolerance only' if pos==0 else None)
            ax.set_xticks(range(len({r['parameters']['n'] for r in initial})),
                          [f'n={n}' for n in sorted({r['parameters']['n'] for r in initial})])
            ax.set_ylabel('Fraction of tasks'); ax.set_ylim(0,1); ax.legend(); fig.tight_layout()
            for ext in ('png','pdf'): fig.savefig(figures / f'initial_stop.{ext}', bbox_inches='tight')
            plt.close(fig)
    if manifest['suite']=='stress':
        stress_groups=defaultdict(list)
        for r in rows:
            if r['model_value'] is not None:
                key=(r.get('scenario','nominal'),r['method'],r['memory'],r.get('assumed_current',''))
                stress_groups[key].append(r['model_value']-r['expected_utility'])
        if stress_groups:
            keys=sorted(stress_groups,key=str)
            fig,ax=plt.subplots(figsize=(8,max(3,len(keys)*.38)))
            ax.barh(range(len(keys)),[float(np.mean(stress_groups[k])) for k in keys])
            ax.set_yticks(range(len(keys)),[' / '.join(str(x) for x in k if x != '') for k in keys])
            ax.axvline(0,color='black',linewidth=.7)
            ax.set_xlabel('Mean model value minus true policy utility')
            fig.tight_layout()
            for ext in ('png','pdf'): fig.savefig(figures / f'model_mismatch.{ext}', bbox_inches='tight')
            plt.close(fig)
    result={'suite':manifest['suite'],'tasks':manifest['task_count'],'records':len(rows),
            'completed_jobs':len(read_jsonl(root / 'jobs.jsonl')),'expected_jobs':manifest['job_count'],
            'config_sha256':manifest['config_sha256'],'planning_source_sha256':manifest['source_sha256'],
            'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'analysis_numpy':np.__version__,
            'comparisons':comparisons, 'timing_note':'Only uncontended single-worker timing is summarized.'}
    dump_json(out / 'coverage.json',result)
    return result
