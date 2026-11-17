"""Exploratory history-mechanism analysis of the unchanged formal benchmark.

No replanning, sample addition, or production changes. Uncertainty intervals are
pointwise, setting-stratified paired bootstrap intervals, not familywise claims.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Also supports direct invocation from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_handoff import estimate, write_csv
from rl_retest.analysis import _rows
from rl_retest.experiments.records import read_jsonl

TOL = 1e-10  # numerical equality, not a practical-equivalence margin


def cell(r):
    return tuple(r[k] for k in ('k','horizon','n','p_c_current','cost'))


def summary(rows, keys):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,items in sorted(groups.items(),key=lambda kv:str(kv[0])):
        values=defaultdict(list)
        for r in items:values[cell(r)].append(r['advantage'])
        e=estimate(values)
        # Every main cell has 20 tasks. For exploratory strata, weight occupied
        # settings equally and expose sample/setting counts rather than pool them.
        def avg(field):
            cells=defaultdict(list)
            for r in items:cells[cell(r)].append(r[field])
            return float(np.mean([np.mean(x) for x in cells.values()]))
        result.append(dict(zip(keys,key))|e|dict(
            terminal_reward_gain=avg('terminal_reward_gain'),test_cost_saving=avg('test_cost_saving'),
            test_count_saving=avg('test_count_saving'),benefit_fraction=avg('benefits'),
            identical_policy_fraction=avg('identical_policy'),root_action_change_fraction=avg('root_action_changed'),
            correct_stop_fraction=avg('correct_stops'),other_stop_fraction=avg('other_stops'),
            descriptive_positive_case_gain=float(np.mean([r['advantage'] for r in items if r['benefits']])) if any(r['benefits'] for r in items) else 0.0))
    return result


def build(source,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest,records=_rows(source)
    tasks={t['task_id']:t for t in read_jsonl(source/'tasks.jsonl')}
    exact=defaultdict(dict)
    for r in records:
        if r['method']=='exact':
            assert r['memory'] not in exact[r['task_id']], 'duplicate exact condition'
            exact[r['task_id']][r['memory']]=r
    assert len(exact)==3240
    rows=[]
    for task_id,rs in exact.items():
        assert set(rs)=={'correct','reset','misread'}
        task=tasks[task_id];p=task['parameters'];a=rs['correct']
        for memory in ('reset','misread'):
            b=rs[memory]
            gain=a['expected_utility']-b['expected_utility']
            reward=manifest['config']['model']['reward']*(a['success_probability']-b['success_probability'])
            tests=b['expected_tests']-a['expected_tests']
            saving=p['cost']*tests
            assert abs(gain-reward-saving)<TOL
            assert gain>=-TOL
            root=lambda x:(x['policy']['action'],x['policy']['path'])
            rows.append(dict(task_id=task_id,**p,memory=memory,advantage=gain,
                terminal_reward_gain=reward,test_cost_saving=saving,test_count_saving=tests,
                benefits=int(gain>TOL),identical_policy=int(a['policy']==b['policy']),
                root_action_changed=int(root(a)!=root(b)),correct_stops=int(root(a)[0]=='stop'),
                other_stops=int(root(b)[0]=='stop'),
                high_A_prior_paths=sum(prior[2]==.7 for prior in task['priors']),
                any_high_A_prior=int(any(prior[2]==.7 for prior in task['priors'])),
                all_success_paths=sum(x==p['n'] for x in task['successes'][0]) if p['n'] else 0,
                any_all_success=int(p['n']>0 and any(x==p['n'] for x in task['successes'][0])),
                total_historical_successes=sum(task['successes'][0]),
                correct_utility=a['expected_utility'],other_utility=b['expected_utility']))
    # Exact diagnostic chosen to explain failed-history reinterpretation; it is
    # separate from the benchmark and carries no sampling confidence interval.
    from fractions import Fraction as F
    prior=(F(1,5),F(3,5),F(1,5));old=(F(0),F(1,10),F(9,10));now=(F(0),F(9,10),F(9,10))
    weights=[b*(1-p)**3 for b,p in zip(prior,old)]
    true=[w/sum(weights) for w in weights]
    wrong_weights=[b*(1-p)**3 for b,p in zip(prior,now)]
    wrong=[w/sum(wrong_weights) for w in wrong_weights]
    mu=sum(b*p for b,p in zip(true,now));wrong_mu=sum(b*p for b,p in zip(wrong,now))
    q=F(1,2);cost=F(1,50)
    success_mu=sum(b*p*p for b,p in zip(true,now))/mu
    failure_mu=sum(b*p*(1-p) for b,p in zip(true,now))/(1-mu)
    test_value=-cost+mu*max(q,success_mu)+(1-mu)*max(q,failure_mu)
    assert success_mu>q and failure_mu<q and test_value>max(q,mu)
    wrong_s=sum(b*p*p for b,p in zip(wrong,now))/wrong_mu
    wrong_f=sum(b*p*(1-p) for b,p in zip(wrong,now))/(1-wrong_mu)
    wrong_test=-cost+wrong_mu*max(q,wrong_s)+(1-wrong_mu)*max(q,wrong_f)
    assert wrong_test<q and wrong_mu<q
    diagnostic=dict(prior=[str(x) for x in prior],history='three failures at old (0,.1,.9)',current='(0,.9,.9)',horizon=1,cost=str(cost),
        correct_posterior=[str(x) for x in true],misread_posterior=[str(x) for x in wrong],
        correct_predictive=str(mu),misread_predictive=str(wrong_mu),correct_test_value=str(test_value),
        correct_policy='test; success -> candidate; failure -> fallback',misread_policy='stop fallback',
        misread_true_value=str(q),correct_minus_misread=str(test_value-q),type='exact illustrative diagnostic, not a sampled benchmark')
    (out/'failure_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    write_csv(out/'task_mechanisms.csv',rows)
    main=summary(rows,['n','memory']);write_csv(out/'overall.csv',main)
    cells=summary([r for r in rows if r['n']==3],['memory','p_c_current','cost'])
    write_csv(out/'capability_cost.csv',cells)
    hist=summary([r for r in rows if r['n']==3],['memory','k','any_all_success'])
    write_csv(out/'history_signal_strata.csv',hist)
    prior=summary([r for r in rows if r['n']==3],['memory','k','any_high_A_prior'])
    write_csv(out/'prior_composition_strata.csv',prior)
    # Cell-level exposure makes exploratory subgroup weighting inspectable.
    write_csv(out/'full_cells.csv',summary(rows,['memory','k','horizon','n','p_c_current','cost']))
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row')
    for i,memory in enumerate(('reset','misread')):
        for j,cost in enumerate((.005,.02,.1)):
            ax=axes[i,j];rs=sorted([r for r in cells if r['memory']==memory and r['cost']==cost],key=lambda r:r['p_c_current'])
            xs=[r['p_c_current'] for r in rs]
            for field,label,marker in [('mean','Total utility advantage','o'),('terminal_reward_gain','Terminal reward gain','s'),('test_cost_saving','Saved test cost','^')]:
                ax.plot(xs,[r[field] for r in rs],marker=marker,label=label)
            ax.fill_between(xs,[r['ci_low'] for r in rs],[r['ci_high'] for r in rs],alpha=.15)
            ax.axhline(0,c='k',lw=.6);ax.set_title(f'Correct - {memory}; cost={cost}')
            ax.set_xticks([.1,.5,.9]);ax.set_xlabel('Current capability probability')
            if j==0:ax.set_ylabel('Expected utility units (R=1)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('History value decomposes into terminal reward and saved testing cost (n=3)')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'history_decomposition.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,memory in zip(axes,('reset','misread')):
        for pc in (.1,.5,.9):
            rs=sorted([r for r in cells if r['memory']==memory and r['p_c_current']==pc],key=lambda r:r['cost'])
            ax.plot([r['cost'] for r in rs],[r['benefit_fraction'] for r in rs],marker='o',label=f'p_C={pc}')
        ax.set_xscale('log');ax.set_ylim(-.02,1.02);ax.set_title(f'Correct - {memory}')
        ax.set_xlabel('Test cost');ax.set_ylabel('Fraction with utility gain > 1e-10');ax.legend()
    fig.suptitle('Where historical interpretation changes decision value (n=3)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'history_benefit_frequency.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    meta={'source':str(source),'source_tasks_sha256':manifest['tasks_sha256'],'source_config_sha256':manifest['config_sha256'],
        'source_planning_sha256':manifest['source_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'analysis':'post hoc exploratory reuse of fixed formal data; no new tasks',
        'bootstrap':{'samples':5000,'seed':20261129,'interval':'pointwise 95% percentile; setting-stratified paired tasks'},
        'numeric_equality_tolerance':TOL,'main_tasks':len(tasks),'paired_condition_rows':len(rows)}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    lines=['# 历史信息价值：机制重分析','',
        '## 本轮问题与口径',
        'A1：reset与misread损失分别在哪些能力/成本条件出现？A2：效用差来自终止成功率还是省下的测试成本？A3：好历史信号或高A先验是否改变解释？',
        '复用固定主网格3240任务；没有新增种子、任务或方法。新增分层与机制解释均属探索性。各完整参数设置等权，设置内对任务做5000次配对bootstrap，报告点态95%区间；并非整个报告的联合95%保证。n=3各记忆比较1620对任务，n=0作为严格模型负对照。',
        '效用恒等分解：ΔJ=RΔP(终止成功)+κ[E(N_other)−E(N_correct)]。本表逐任务数值残差小于1e-10；该恒等式不需要额外抽样置信区间。',
        '收益发生频率的阈值1e-10仅区分数值零，不是事后选择的“实用收益”阈值；正收益条件均值只描述已观察子群，不能替代全体效应。','',
        '## 核心数量']
    for r in main:
        lines.append(f"- n={r['n']}, correct−{r['memory']}: 平均{r['mean']:.8f}, 95% CI [{r['ci_low']:.8f},{r['ci_high']:.8f}]；正收益比例{r['benefit_fraction']:.2%}；完整策略相同比例{r['identical_policy_fraction']:.2%}；根动作改变比例{r['root_action_change_fraction']:.2%}。")
    lines+=['','## 按能力与成本的效用分解','|对照|当前p_C|成本|总效用差|终止奖励差|测试成本节省|正收益比例|','|---|---:|---:|---:|---:|---:|---:|']
    for r in cells:lines.append(f"|{r['memory']}|{r['p_c_current']}|{r['cost']}|{r['mean']:.6f}|{r['terminal_reward_gain']:.6f}|{r['test_cost_saving']:.6f}|{r['benefit_fraction']:.1%}|")
    lines+=['','## 结论—证据—决策',
        '- 已支持（理论+负对照）：能力不变时correct与misread相同，无历史时三条件相同。负对照中的零区间不是小样本证明等效，而来自同一模型与策略问题。',
        '- 限定条件下支持：正确解释历史的增益高度依赖比较对象、能力与成本。见capability_cost.csv；不写“历史总能显著提升”。对同一实际历史，正确后验最优策略不劣于其他可执行策略是策略最优性；不能把不同信念下V(b_H)与V(b_0)当逐历史信息价值序关系。',
        '- 机制检验与修正：p_C=.5、κ=.1时correct和reset都100%立即停止；平均增益.131968全部来自终止选路，省测试项为零。因此删除“高成本历史收益主要来自省测试”的概括。相对misread，在p_C=.9三个成本下的省测试项反而为负，说明正确历史有时值得多测，以换取更好的终止选择。',
        '- 误读会产生两种相反错误：p_C=.5、κ=.005时，correct在99.44%任务立即停止，misread仅65%；总差.00247684中.00206130来自节约测试成本。p_C=.9时，misread在全部任务立即停止，而correct仍在5.6%–7.8%任务选择测试；此时终止奖励改善超过额外测试成本。报告应分别解释无益重测和过早放弃，不能把误读归为单一种类错误。',
        '- 行动差异不等于效用差：p_C=.9=p_A时，任一路径历史有成功即排除I；C和A的当前执行率相同。不同路径/类型后验可能产生相同效用，故不能把根动作改变率当作增益发生率。',
        '- 探索性分层：history_signal_strata.csv按K及是否有三次全成功路径分组；prior_composition_strata.csv按K及是否存在A质量.7的先验分组。子群大小不均，按占据的原设置等权；这些关联不隔离所有混杂因素，不能写成先验组成的独立因果效果。',
        '- 决策：报告主文保留全体均值及能力/成本分解图，历史信号和先验分层放附录。没有按显著性追加样本。该轮解决“收益来自什么”的问题；更细的先验因果敏感性不改变课程主结论，停止扩展。','',
        '## 统计与展示说明',
        '- history_decomposition.pdf/png：横轴当前C类型成功率；纵轴R=1效用单位；实线总差、终止奖励差、节约成本，带状区间只对应总效用差。两行reset/misread，三列成本；全部为n=3。',
        '- history_benefit_frequency.pdf/png：横轴测试成本（对数），纵轴正收益任务比例；颜色为当前能力。比例是设置等权描述性统计，不附推断置信区间。',
        '- capability_cost.csv和full_cells.csv可追溯全部分面；task_mechanisms.csv保存每任务配对及先验/历史特征，供复查。',
        '- 复现：`PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python scripts/analyze_history.py`。生产模型和旧原始结果均未修改。']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:15]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('artifacts/final-20261117/main'));p.add_argument('--out',type=Path,default=Path('artifacts/research-v2-20261117/history'))
    a=p.parse_args();build(a.source,a.out)
