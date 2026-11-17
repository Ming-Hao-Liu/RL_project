"""Exploratory quality/cost/certificate decomposition of the frozen P0 study."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.analysis import _rows,_cell
from rl_retest.experiments.records import read_json,read_jsonl,policy_to_data
from rl_retest.experiments.tasks import Task,contexts
from rl_retest.planners.common import TransitionCache
from rl_retest.planners.dynamic import _Bellman
from scripts.build_handoff import write_csv,estimate

ROOT=Path('artifacts/final-20261117');OUT=Path('artifacts/research-v2-20261117/search')

def save(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    main_manifest,rows=_rows(ROOT/'main')
    exact={r['task_id']:r for r in rows if r['method']=='exact' and r['memory']=='correct'}
    quality=[]
    for h in (1,2,3):
        for method in ('myopic','depth2','blinkered','bounded','iterative'):
            rr=[r for r in rows if r['method']==method and r['memory']=='correct' and r['parameters']['horizon']==h]
            losses=np.array([exact[r['task_id']]['expected_utility']-r['expected_utility'] for r in rr])
            assert losses.min()>-1e-10
            widths=[r['upper']-r['lower'] for r in rr if r['upper'] is not None]
            quality.append(dict(horizon=h,method=method,tasks=len(rr),exact_value_fraction=float(np.mean(losses<=1e-10)),mean_regret=float(losses.mean()),max_regret=float(losses.max()),mean_width=float(np.mean(widths)) if widths else '',exact_but_positive_width=sum(loss<=1e-10 and r['upper'] is not None and r['upper']-r['lower']>1e-10 for loss,r in zip(losses,rr))))
    write_csv(OUT/'main_quality.csv',quality)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for method in ('myopic','depth2','blinkered','bounded','iterative'):
        rs=[r for r in quality if r['method']==method]
        axs[0].plot([r['horizon'] for r in rs],[r['exact_value_fraction'] for r in rs],'-o',label=method)
        axs[1].plot([r['horizon'] for r in rs],[r['mean_regret'] for r in rs],'-o',label=method)
    axs[0].set_ylim(0,1.02);axs[0].set_ylabel('Fraction attaining exact value (tolerance 1e-10)');axs[1].set_ylabel('Mean exact regret');axs[1].legend()
    for ax in axs:ax.set_xlabel('Test horizon B');ax.set_xticks([1,2,3])
    save(fig,'shallow_sufficiency')
    manifest,large=_rows(ROOT/'large')
    records=[];summary=[];convergence=[]
    for n in (0,3):
        selected=[r for r in large if r['parameters']['n']==n]
        for method in ('stop','myopic','blinkered','bounded','iterative'):
            for budget in ((100,1000,10000) if method in ('bounded','iterative') else (None,)):
                rr=[r for r in selected if r['method']==method and r.get('budget')==budget and (method!='bounded' or r['epsilon']==.01)]
                summary.append(dict(n=n,method=method,budget=budget,tasks=len(rr),mean_utility=np.mean([r['expected_utility'] for r in rr]),mean_evaluations=np.mean([r['evaluations'] for r in rr]),median_seconds=np.median([r['planning_seconds'] for r in rr]),p90_seconds=np.quantile([r['planning_seconds'] for r in rr],.9),mean_scans=np.mean([r['action_scans'] for r in rr]),mean_width=np.mean([r['upper']-r['lower'] for r in rr]) if method=='bounded' else ''))
    write_csv(OUT/'natural_cost.csv',summary)
    policies={}
    for r in large:
        if r['method'] in ('myopic','blinkered'):policies.setdefault(r['task_id'],{})[r['method']]=r
    for r in large:
        if r['method']!='bounded' or r['epsilon']!=.01:continue
        baseline=policies[r['task_id']]['myopic']
        assert r['upper']+1e-10>=baseline['model_value']
        records.append(dict(task_id=r['task_id'],**r['parameters'],budget=r['budget'],lower=r['lower'],upper=r['upper'],initial_upper=max(r['initial_stop'],r['perfect_information']-r['parameters']['cost']),baseline_value=baseline['model_value'],baseline_gap_upper=r['upper']-baseline['model_value'],search_gap_upper=r['upper']-r['lower'],baseline_evaluations=baseline['evaluations'],search_evaluations=r['evaluations']))
    write_csv(OUT/'certificate_tasks.csv',records)
    for n in (0,3):
        for m in (100,1000,10000):
            rr=[r for r in records if r['n']==n and r['budget']==m]
            convergence.append(dict(n=n,budget=m,mean_lower=np.mean([r['lower'] for r in rr]),mean_upper=np.mean([r['upper'] for r in rr]),mean_baseline=np.mean([r['baseline_value'] for r in rr]),mean_initial_upper=np.mean([r['initial_upper'] for r in rr]),mean_baseline_gap_upper=np.mean([r['baseline_gap_upper'] for r in rr])))
    write_csv(OUT/'certificate_convergence.csv',convergence)
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for n,ax in zip((0,3),axs):
        rr=[r for r in convergence if r['n']==n];x=[r['budget'] for r in rr]
        for key,label in [('mean_lower','Search policy L'),('mean_upper','Search upper U'),('mean_baseline','Myopic feasible value'),('mean_initial_upper','Initial upper')]:ax.plot(x,[r[key] for r in rr],'-o',label=label)
        ax.set_xscale('log');ax.set_xlabel('Successor-belief budget M');ax.set_ylabel('Mean utility / model bound');ax.set_title(f'n={n}')
    axs[1].text(.04,.05,'All four curves coincide: stop certified',transform=axs[1].transAxes,fontsize=9);axs[1].legend();save(fig,'certificate_decomposition')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in ('stop','myopic','blinkered','bounded','iterative'):
        rr=[r for r in summary if r['n']==0 and r['method']==method]
        axs[0].plot([max(1,r['mean_evaluations']) for r in rr],[r['mean_utility'] for r in rr],'-o',label=method)
        axs[1].plot([r['median_seconds'] for r in rr],[r['mean_utility'] for r in rr],'-o',label=method)
    for ax in axs:ax.set_xscale('log');ax.set_ylabel('Mean utility, n=0')
    axs[0].set_xlabel('Mean successor computations (STOP shown at 1)');axs[1].set_xlabel('Median observed single-worker seconds');axs[1].legend();save(fig,'quality_cost')
    # Predetermined implementation diagnostic: first K=20/40, B=6, n=0 tasks.
    protocol={'question':'Does full-depth completion explain iterative budget plateaus?', 'selection':'index=0,K in [20,40],B=6,n=0 from frozen large tasks','depths':[1,2],'expected_computations':'d1=2K; d2=2K+(2K)^2 for strictly positive outcome branches','classification':'deterministic implementation diagnostic, no population inference'}
    (OUT/'diagnostic_protocol.json').write_text(json.dumps(protocol,indent=2))
    diagnostic=[]
    for data in read_jsonl(ROOT/'large/tasks.jsonl'):
        p=data['parameters']
        if p['index']!=0 or p['horizon']!=6 or p['n']!=0:continue
        task=Task.from_data(data);problem,_=contexts(task,manifest['config']);cache=TransitionCache(problem);bellman=_Bellman(problem,cache)
        for depth in (1,2):
            val,policy=bellman.solve((0,)*(2*p['k']),problem.belief,depth)
            expected=2*p['k']+(2*p['k'])**2 if depth==2 else 2*p['k']
            assert cache.stats.evaluations==expected
            saved=[r for r in large if r['task_id']==task.task_id and r['method']=='iterative' and r['budget']==(1000 if depth==1 else 10000)][0]
            assert policy_to_data(policy)==saved['policy']
            diagnostic.append(dict(k=p['k'],depth=depth,computations=cache.stats.evaluations,model_value=val,saved_policy_match=True))
    write_csv(OUT/'depth_completion.csv',diagnostic)
    plateau=[]
    for n in (0,3):
        ids={r['task_id'] for r in large if r['parameters']['n']==n}
        equal=0
        for tid in ids:
            rr={r['budget']:r for r in large if r['task_id']==tid and r['method']=='iterative'}
            equal+=rr[100]['policy']==rr[1000]['policy']
        plateau.append(dict(n=n,tasks=len(ids),identical_policy_M100_M1000=equal))
    write_csv(OUT/'iterative_plateau.csv',plateau)
    _,stress=_rows(ROOT/'stress');nom={r['task_id']:r for r in stress if r.get('scenario')=='nominal'}
    stability=[]
    for n in (0,3):
        for pc in (.4,.6):
            rr=[r for r in stress if r.get('scenario')=='probability_shift' and r['method']=='exact' and r['parameters']['n']==n and r['assumed_current']==pc]
            stability.append(dict(n=n,assumed_current=pc,tasks=len(rr),identical_policy=sum(r['policy']==nom[r['task_id']]['policy'] for r in rr),same_value=sum(abs(r['expected_utility']-nom[r['task_id']]['expected_utility'])<1e-10 for r in rr)))
    write_csv(OUT/'stress_policy_stability.csv',stability)
    lines=['# 搜索机制：质量、成本、证书分离','\n本分析复用完整原始数据；新分层为探索性，数值阈值1e-10只用于舍入识别。没有新种子确认性试验或新算法。', '\n## 浅层是否足够']
    lines += [str(r) for r in quality if r['horizon']==3]
    lines += ['\n## 迭代加深台阶','预算计量按转移边(state,path,outcome)，不按唯一后继状态。n=0时两种观测均为正概率，完整深度1需要2K次；深度2累计2K+(2K)^2次。K=20时为40/1640；K=40时为80/6480。M=1000无法完成第二层，必须返回上一完整层；M=10000完成第二层。具体诊断与保存策略一致，不能把未完成层的计算当免费。',str(plateau),'\n## 证书的额外价值及成本','同一正确模型下，上界U适用于任何可执行策略。因此U−J(myopic)也可形成该基线的后验质量上界，但这里的U来自额外搜索，必须支付搜索与基线的组合成本；这只是证据分析，不是免费证书或新的优胜算法。大规模没有真实最优值，不把这个上界说成实际regret。',str(convergence),'\n## 压力测试行动稳定性',str(stability),'\n## 研究循环','实际支持：相同M比较与自然成本比较回答不同问题；迭代加深台阶源于完整层交付约束；浅规划充分性可由主任务精确值量化。尚不能从大规模松界判定真实regret大，也不能证明初始上界改进必有净收益。决策：不引入新搜索变体，保留负面排名；报告同时呈现效用、计算次数与上下界。原耗时为单进程观测且含初始化，不据此声称隔离环境速度优势。']
    if (OUT/'tolerance_diagnostic.csv').exists():
        import csv
        diagnostic_rows=list(csv.DictReader((OUT/'tolerance_diagnostic.csv').open()))
        closed=sum(abs(float(r['new_value'])-float(r['exact_value']))<=1e-10 for r in diagnostic_rows)
        lines += ['\n## 容差诊断：证据与决策',f'原主网格全部{len(diagnostic_rows)}个非精确搜索结果为正容差停止；固定M=10000、只改epsilon=0后，{closed}个达到精确值。平均更新次数从27.96到253.88。协议、完整策略和结果见tolerance_protocol.json、tolerance_policies.jsonl、tolerance_diagnostic.csv。此为已有任务上的探索性干预，不代替原主表，也不推广为大规模必能低成本求精确最优。结论：小规模已观察损失来自容差取舍，结束此诊断。']
    (OUT/'SUMMARY.md').write_text('\n\n'.join(lines)+'\n')

if __name__=='__main__':main()
