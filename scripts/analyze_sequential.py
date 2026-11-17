"""Exploratory decomposition of the frozen benchmark and exact two-test diagnosis."""
import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl_retest.analysis import _rows, _cell
from rl_retest.model import ModelSpec, PlanningProblem, EvaluationContext
from rl_retest.planners.dynamic import solve_exact, solve_lookahead
from rl_retest.evaluation import evaluate
from rl_retest.experiments.records import policy_to_data
from scripts.build_handoff import estimate, write_csv

TOL=1e-10


def summarize(rows,group,label,ci=True):
    values=np.array([r['gain'] for r in rows]); positive=values>TOL
    cells=defaultdict(list)
    for r in rows:cells[tuple(r[k] for k in ('k','horizon','n','p_c_current','cost'))].append(r['gain'])
    est=estimate(cells) if ci else {}
    mean=float(values.mean());conditional=float(values[positive].mean()) if positive.any() else 0.
    return dict(group=group,label=label,tasks=len(rows),positive=int(positive.sum()),frequency=float(positive.mean()),mean_gain=mean,conditional_gain=conditional,frequency_times_conditional=float(positive.mean())*conditional,residual=mean-float(positive.mean())*conditional,max_gain=float(values.max()),ci_low=est.get('ci_low',''),ci_high=est.get('ci_high',''),myopic_stop=sum(r['myopic_root']=='stop' for r in rows),exact_stop=sum(r['exact_root']=='stop' for r in rows),mean_exact_tests=float(np.mean([r['exact_tests'] for r in rows])),mean_myopic_tests=float(np.mean([r['myopic_tests'] for r in rows])),mean_terminal_reward_gain=float(np.mean([r['terminal_reward_gain'] for r in rows])),mean_extra_test_cost=float(np.mean([r['extra_expected_test_cost'] for r in rows])))


def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    cstar=F(91,11600)
    costs=[0.,.001,.003,.005,.007,.0078,float(cstar),.0079,.008,.01,.02]
    protocol={'classification':'existing benchmark reanalysis is exploratory; analytic cost sweep is a diagnostic, not a new sampled confirmation','question':'Is the small mean DP-minus-rolling-myopic gain caused by rare gains or small conditional gains? What cost boundary permits two tests when one test has no benefit?','source':str(source),'numerical_positive_threshold':TOL,'threshold_meaning':'float error classification, not a minimum meaningful effect','bootstrap_samples':5000,'bootstrap_seed':20261129,'fixed_diagnostic':{'prior':[0,.05,.95],'probabilities':[0,.8,.05],'fallback':.5,'reward':1,'horizon':2,'costs':costs,'analytic_boundary':str(cstar)},'stop_rule':'analyze all 3240 existing pairs and all 11 declared cost points once; no sample extension'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    manifest,records=_rows(source)
    tasks={r['task_id']:r for r in map(json.loads,(source/'tasks.jsonl').read_text().splitlines())}
    selected=defaultdict(dict)
    for r in records:
        if r['memory']=='correct' and r['method'] in ('exact','myopic'):
            assert r['method'] not in selected[r['task_id']]
            selected[r['task_id']][r['method']]=r
    assert len(selected)==3240 and all(set(v)=={'exact','myopic'} for v in selected.values())
    rows=[]
    for tid,v in selected.items():
        a,b=v['exact'],v['myopic'];gain=a['expected_utility']-b['expected_utility']
        assert gain>=-TOL
        ar,br=a['policy']['action'],b['policy']['action']
        kind=('both_stop' if ar==br=='stop' else 'myopic_stop_exact_test' if ar=='test' and br=='stop' else 'both_test_same_root' if ar==br=='test' and a['policy']['path']==b['policy']['path'] else 'both_test_different_root' if ar==br=='test' else 'exact_stop_myopic_test')
        prior_count=sum(p[2]==.7 for p in tasks[tid]['priors'])
        rows.append(dict(task_id=tid,**a['parameters'],gain=gain,positive=gain>TOL,exact_root=ar,myopic_root=br,root_class=kind,prior_A_rich_count=prior_count,has_A_rich_prior=bool(prior_count),initial_stop=a['initial_stop'],perfect_information=a['perfect_information'],exact_tests=a['expected_tests'],myopic_tests=b['expected_tests'],terminal_reward_gain=a['success_probability']-b['success_probability'],extra_expected_test_cost=a['parameters']['cost']*(a['expected_tests']-b['expected_tests'])))
    write_csv(out/'task_decomposition.csv',rows)
    summaries=[summarize(rows,'all','all')]
    for field in ('horizon','cost','n','p_c_current','has_A_rich_prior','root_class'):
        for key in sorted({r[field] for r in rows},key=str):
            subset=[r for r in rows if r[field]==key]
            # Outcome-conditioned groups are descriptive; no post-selection confidence claim.
            summaries.append(summarize(subset,field,str(key),ci=field in ('horizon','cost','n','p_c_current')))
    for b in (1,2,3):
        for c in (.005,.02,.1):
            summaries.append(summarize([r for r in rows if r['horizon']==b and r['cost']==c],'horizon_cost',f'B={b}, cost={c}'))
    assert max(abs(r['residual']) for r in summaries)<TOL
    assert max(abs(r['gain']-r['terminal_reward_gain']+r['extra_expected_test_cost']) for r in rows)<TOL
    write_csv(out/'decomposition.csv',summaries)
    figs=[]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    bs=[r for r in summaries if r['group']=='horizon']
    for ax,key,title in zip(axes,('mean_gain','frequency','conditional_gain'),('All-task mean gain','Fraction with gain > 1e-10','Mean gain among positive tasks')):
        errors = None
        if key == 'mean_gain':
            errors = [[r[key] - r['ci_low'] for r in bs],
                      [r['ci_high'] - r[key] for r in bs]]
            title += '\n(pointwise 95% paired bootstrap CI)'
        ax.bar([r['label'] for r in bs],[r[key] for r in bs],color='#356b9b',yerr=errors,capsize=4)
        ax.set_xlabel('Test horizon B');ax.set_title(title);ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Expected utility (R=1)');axes[2].set_ylabel('Expected utility (R=1)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'gain_decomposition.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for b in (2,3):
        rs=[r for r in summaries if r['group']=='horizon_cost' and r['label'].startswith(f'B={b},')]
        x=[.005,.02,.1]
        axes[0].plot(x,[r['frequency'] for r in rs],'-o',label=f'B={b}');axes[1].plot(x,[r['mean_gain'] for r in rs],'-o',label=f'B={b}')
    for ax in axes:ax.set_xlabel('Test cost');ax.set_ylim(bottom=0);ax.legend()
    axes[0].set_ylabel('Fraction with gain > 1e-10');axes[1].set_ylabel('All-task mean utility gain');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'cost_decomposition.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    # Exact posterior table: type is drawn once, and likelihood accumulates over tests.
    posterior=[]
    for obs in ('','F','S','FF','FS','SF','SS'):
        wc=F(1,20);wa=F(19,20)
        for o in obs:
            wc*=F(4,5) if o=='S' else F(1,5);wa*=F(1,20) if o=='S' else F(19,20)
        mass=wc+wa;pc=wc/mass;prediction=pc*F(4,5)+(1-pc)*F(1,20)
        posterior.append(dict(observations=obs or 'root',sequence_probability=float(mass),posterior_C=float(pc),predictive_success=float(prediction),predictive_fraction=str(prediction),best_terminal='candidate' if prediction>F(1,2) else 'fallback',optimal_action_at_cost_005='test' if obs in ('','S') else 'stop',reachable_under_optimal_policy=obs in ('','F','S','SF','SS')))
    write_csv(out/'analytic_decision_table.csv',posterior)
    diagnostics=[]
    for c in costs:
        problem=PlanningProblem(((0,.05,.95),),ModelSpec(1,c,2,.5,(0,.8,.05),()))
        exact=solve_exact(problem);myopic=solve_lookahead(problem,1)
        analytic=max(.5,.5+float(F(273,32000))-float(F(87,80))*c)
        evaluation=evaluate(exact.policy,EvaluationContext(problem.belief,problem.spec))
        assert abs(exact.model_value-analytic)<TOL and abs(evaluation.expected_utility-analytic)<TOL and abs(myopic.model_value-.5)<TOL
        diagnostics.append(dict(cost=c,analytic_value=analytic,exact_value=exact.model_value,myopic_value=myopic.model_value,expected_tests=evaluation.expected_tests,policy=json.dumps(policy_to_data(exact.policy),sort_keys=True)))
    write_csv(out/'cost_diagnostic.csv',diagnostics)
    fig,ax=plt.subplots(figsize=(7,3.8))
    ax.plot(costs,[r['exact_value']-.5 for r in diagnostics],'-o',label='Exact two-test gain')
    ax.plot(costs,[r['myopic_value']-.5 for r in diagnostics],'--',label='Rolling myopic gain')
    ax.axvline(float(cstar),ls=':',c='k',label='Analytic threshold 91/11600');ax.set_xlabel('Test cost');ax.set_ylabel('Expected utility above fallback (R=1)');ax.set_ylim(bottom=-.0005);ax.legend();fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'analytic_cost_boundary.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    best=max(rows,key=lambda r:r['gain']);sample=selected[best['task_id']]
    (out/'representative_benchmark.json').write_text(json.dumps({'selection':'maximum observed DP-minus-myopic gain; exploratory illustration, not representative sampling','task':tasks[best['task_id']],'decomposition':best,'exact_record':sample['exact'],'myopic_record':sample['myopic']},indent=2))
    (out/'provenance.json').write_text(json.dumps({'source_manifest':manifest,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_records':len(records),'paired_tasks':len(rows),'checks':{'decomposition_residual_lt_1e_10':True,'reward_minus_cost_identity':True,'analytic_DP_evaluation_agreement':True}},indent=2))
    overall=summaries[0]
    text=['# 连续测试：机制重分析与解析成本边界','\n## 证据口径','所有 3240 个主任务均保留；exact/myopic 是同一真实条件模型下的完整冻结策略。myopic 每次观测后重新做一步前瞻，其实现不是“最多测一次”。当前分析为已有数据的探索性重分析；正收益阈值1e-10仅排除浮点误差，不是实际有意义差异阈值。无正收益任务时conditional_gain记0仅为分解占位，不是存在一个条件样本的均值估计。按B、成本、n和能力的分组仍含每设置20任务，均值与设置等权一致；根动作/先验构成分组仅报告条件描述，不作确认性推断。','95%区间为5000次设置内配对bootstrap，使用原定bootstrap种子20261129。它们是逐比较区间，不是多个图表联合95%保证。条件正收益均值不附上普通bootstrap区间，以免把按结果筛选子群当确认性结果。','\n## 主要结果',f"全体均值 {overall['mean_gain']:.10f} = 正收益频率 {overall['frequency']:.6%} × 条件正收益 {overall['conditional_gain']:.10f}（残差 {overall['residual']:.2g}）。"]
    for r in bs:text.append(f"- B={r['label']}: {r['positive']}/{r['tasks']} 个任务正收益，频率 {r['frequency']:.4%}，全体均值 {r['mean_gain']:.10f}，95% CI [{r['ci_low']:.10f}, {r['ci_high']:.10f}]，条件收益 {r['conditional_gain']:.10f}，最大 {r['max_gain']:.10f}。")
    text+=['\n### 根动作与成本解释']
    for r in summaries:
        if r['group'] in ('root_class','cost','n','has_A_rich_prior'):
            text.append(f"- {r['group']}={r['label']}: N={r['tasks']}, 正收益={r['positive']}, 均值={r['mean_gain']:.10f}, 条件收益={r['conditional_gain']:.10f}。")
    text += ['\n主网格正收益的161任务中，仅9例是myopic根节点停止而DP开始测试；34例双方首测路径不同，118例首测相同但后续不同。故解析反例说明存在性，不代表主要经验增益都来自突破首测门槛。成本.02组DP平均测试次数反而更少：更深规划还可通过避免不值得的后续测试提高净效用，不等于总是多测。先验A-rich组出现更多正收益的描述性关联，不支持“只要有一个A-rich先验就无需多步规划”；这还混合了路径数、能力和历史因素。']
    text+=['\n## 解析过程与边界','候选路径先验P(C)=1/20，P(A)=19/20；当前成功率4/5与1/20；fallback=1/2。初始预测7/80；一次成功后的预测11/28≈.392857仍低于fallback，所以单步信息无法改变终止选择。两次成功后的预测823/1100≈.748182高于fallback。一次失败后再次成功的预测277/1700≈.162941仍低于fallback，所以最优两步策略是首测失败停止、成功才再测、只有SS选候选。','该策略两次成功的概率11/320；终止奖励改善273/32000；期望测试数87/80。因此 V₂(c)=1/2+max(0,273/32000−87c/80)，c*=91/11600≈.0078448276。所有非负成本下V₁(c)=1/2。c=.005时V₂=.50309375；c≥c*停止最优（等号时允许平局）。11个预定成本点均通过DP与固定策略评价核验，误差<1e-10。数值不需要采样区间。','先验与fallback边界也可解析给出：保持p_C=.8,p_A=.05,q=.5，令r=P(C)，一次成功后仍不优于fallback而两次成功后优于fallback的区间为3/515<r≤3/35。该区间内两次成功改选路径的收益为D(r)=.193125r−.001125，策略期望测试数为1.05+.75r，正净收益需要c<D(r)/(1.05+.75r)。固定r=.05时，fallback满足11/28≤q<823/1100即可保留这一单步/两步分界；相应临界成本为(823/32000−11q/320)/(87/80)。这些是固定两类型、两步模型的解析性质，不向原始全部任务推广。','该例单独改变测试成本，其他因素固定，给出明确的机制证据；它没有证明原始随机分布中的典型增益。完整后验与决策表见analytic_decision_table.csv。','\n## 结论—证据—决策','- 已支持（解析存在性）：单次测试无利而连续测试有利。依据上式与逐步后验，不需要增加随机样本；结束这一问题。','- 限定条件下支持（分布均值）：原主网格DP平均优于myopic但增益量级小。频率×条件收益拆分与根动作分组回答“稀少还是幅度小”；不能改写为一般任务中序贯测试普遍不重要。','- 探索解释：成本与已有有价值路径改变继续测试边界；先验构成关联可能同时受K、能力和历史影响，不能单凭分组因果归因。成本诊断单独验证成本边界，其余不扩展笛卡尔积。','- 报告修改：将“多步规划明显提升性能”收窄为“解析例子证明存在价值；完整分布重分析量化频率与条件幅度”。新增gain_decomposition和analytic_cost_boundary，保留原主实验总体曲线。','- 剩余候选工作：新增更大样本不改变存在性证明，也不是解释原始均值所需；不追加样本。若将来主张普遍的实际收益，必须先规定实际有意义阈值并另设确认分布；当前不作该主张。','\n## 图表与复现','- gain_decomposition.pdf/png：横轴B；三幅纵轴依次为总体净效用差、正收益频率、条件净效用差；数据decomposition.csv。只展示全部任务与条件任务并列，不用条件图替代总体。','- cost_decomposition.pdf/png：横轴测试成本；正收益频率与总体净效用差；基于原始全部网格，探索性。','- analytic_cost_boundary.pdf/png：横轴诊断成本，纵轴相对fallback效用；固定解析例子，不添加采样区间。','- 命令：`PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -m scripts.analyze_sequential`。']
    (out/'SUMMARY.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'overall':overall,'by_horizon':bs,'root_classes':[r for r in summaries if r['group']=='root_class']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('artifacts/final-20261117/main'));p.add_argument('--out',type=Path,default=Path('artifacts/research-v2-20261117/sequential'));a=p.parse_args();main(a.source,a.out)
