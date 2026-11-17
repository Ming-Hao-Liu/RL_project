"""Assemble and verify v2 without changing v1 evidence or rerunning experiments."""
from pathlib import Path
import csv,hashlib,json,shutil,zipfile,re
ROOT=Path.cwd();OUT=ROOT/'artifacts/handoff-v2-20261117';ANALYSIS=ROOT/'artifacts/research-v2-20261117'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return list(csv.DictReader(path.open()))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mapping={'report_v2.md':'REPORT.md','theory.md':'THEORY.md','results_v2.md':'RESULTS.md','ppt_handoff_v2.md':'PPT_HANDOFF.md','defense_qa_v2.md':'DEFENSE_QA.md','technical_review_v2.md':'TECHNICAL_REVIEW.md','report_review_v2.md':'REPORT_REVIEW.md','acceptance_v2.md':'ACCEPTANCE.md','contributions.md':'CONTRIBUTIONS.md'}
    for src,dst in mapping.items():shutil.copy2(ROOT/'docs'/src,OUT/dst)
    shutil.copy2(ROOT/'EVIDENCE_LEDGER.md',OUT/'EVIDENCE_LEDGER.md')
    for sub in ('history','sequential','search','technical'):shutil.copytree(ANALYSIS/sub,OUT/sub,dirs_exist_ok=True)
    for suite in ('main','large','stress'):shutil.copytree(ROOT/'artifacts/final-20261117'/suite,OUT/'raw'/suite,dirs_exist_ok=True)
    for kind in ('figures','tables'):shutil.copytree(ROOT/'artifacts/handoff-20261119'/kind,OUT/'baseline'/kind,dirs_exist_ok=True)
    for sub in ('src','tests','scripts','configs','docs'):
        shutil.copytree(ROOT/sub,OUT/'source'/sub,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml','uv.lock','.python-version','AGENTS.md','Proposal.md','Project_Requirements.md','Guideline.md','EVIDENCE_LEDGER.md','RL_PROJECT_RESEARCH_SPRINT.md'):
        shutil.copy2(ROOT/name,OUT/'source'/name)
    shutil.copy2(ROOT/'README.md',OUT/'source/README.repository.md')
    (OUT/'source/README.md').write_text('# 源码快照\n\n安装、数据布局和复现命令见上一级 README.md。README.repository.md 是原工作区导航快照，其历史artifact链接按原仓库布局解释。\n')
    (OUT/'provenance').mkdir(exist_ok=True)
    old=ROOT/'artifacts/handoff-20261119.zip'
    assert sha(old)==(old.with_suffix('.zip.sha256')).read_text().split()[0]
    for file in (old,old.with_suffix('.zip.sha256')):shutil.copy2(file,OUT/'provenance'/file.name)
    checks={'old_archive_unchanged':True,'old_archive_sha256':sha(old),'raw_suites':{},'review_versions':{}}
    expected={'main':(3240,22140,32400),'large':(160,800,1920),'stress':(40,320,320)}
    for suite,(nt,nj,nr) in expected.items():
        root=OUT/'raw'/suite;m=json.loads((root/'manifest.json').read_text());v=json.loads((root/'verification.json').read_text())
        record_count=sum(len(f.read_text().splitlines()) for f in (root/'records').glob('*.jsonl'))
        assert (m['task_count'],m['job_count'],record_count)==(nt,nj,nr) and v['valid']
        assert sha(root/'tasks.jsonl')==m['tasks_sha256'] and sha(root/'jobs.jsonl')==m['jobs_sha256']
        checks['raw_suites'][suite]={'tasks':nt,'jobs':nj,'records':nr,'verification':v,'source_sha256':m['source_sha256']}
    diag=rows(OUT/'search/tolerance_diagnostic.csv')
    assert len(diag)==199 and all(abs(float(r['new_value'])-float(r['exact_value']))<=1e-10 and r['old_status']=='tolerance' and r['new_status']=='complete' for r in diag)
    assert len(rows(OUT/'sequential/task_decomposition.csv'))==3240
    assert len(rows(OUT/'history/task_mechanisms.csv'))==6480
    checks['mechanism_records']={'history_pairs':6480,'sequential_pairs':3240,'tolerance_pairs':199}
    # All explicit core visual paths in PPT handoff must have vector and preview forms.
    text=(OUT/'PPT_HANDOFF.md').read_text()
    visual=re.findall(r'`((?:history|sequential|search|baseline/figures)/[^`]+\.pdf)`',text)
    for name in visual:
        assert (OUT/name).is_file(),name
        assert (OUT/name).with_suffix('.png').is_file(),name
    checks['ppt_core_figures_checked']=visual
    review=(OUT/'REPORT_REVIEW.md').read_text()
    for file,name in ((OUT/'REPORT.md','report_v2.md'),(OUT/'EVIDENCE_LEDGER.md','EVIDENCE_LEDGER.md')):
        assert sha(file) in review, f'review stale: {name}'
        checks['review_versions'][name]=sha(file)
    checks['scope']='File/count/source/review/diagnostic consistency; scientific acceptance is documented separately in ACCEPTANCE.md and the evidence ledger.'
    (OUT/'package_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    (OUT/'README.md').write_text('''# 第二版研究成果包

交付日期：2026-11-17，截止2026-11-20 00:00 Asia/Singapore。先读 REPORT.md，再用 PPT_HANDOFF.md 选图；RESULTS.md是结论索引，EVIDENCE_LEDGER.md记录每轮证据判断与停止依据，DEFENSE_QA.md用于成员理解。ACCEPTANCE、TECHNICAL_REVIEW、REPORT_REVIEW记录验收与已修正问题。

history/、sequential/、search/提供机制数据、协议、PDF/PNG及解释；raw/保留所有原正式记录；baseline/提供原完整基准图表和表格。provenance/内是未修改的原成果包及摘要，source/是本次完整代码、配置、理论和测试快照。没有用诊断结果替换原方法排名。

## 安装与验证

在source目录执行。所有环境和缓存留在包内；需要uv和Python3.11。

```sh
mkdir -p .cache/uv .cache/python .cache/matplotlib .tmp artifacts
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --locked --group dev
uv run pytest
uv run rl-retest verify --run-dir ../raw/main
uv run rl-retest verify --run-dir ../raw/large
uv run rl-retest verify --run-dir ../raw/stress
```

## 重新生成第二版分析

先建立包内数据链接；以下命令面向macOS/Linux，路径不依赖原机器。

```sh
ln -s ../../raw artifacts/final-20261117
uv run python -m scripts.analyze_history
uv run python -m scripts.analyze_sequential
uv run python -m scripts.diagnose_tolerance
uv run python -m scripts.analyze_search
```

新输出位于source/artifacts/research-v2-20261117/。前三项分别验证历史分解、11点成本边界和全部199个ε干预；最后一项生成质量/成本/证书图。诊断是已有任务上的探索性机制研究，不是独立确认样本。

若需重现技术审阅的旧包验证，先展开旧包并放置其归档引用：

```sh
uv run python -m zipfile -e ../provenance/handoff-20261119.zip artifacts
ln -s ../../provenance/handoff-20261119.zip artifacts/handoff-20261119.zip
ln -s ../../provenance/handoff-20261119.zip.sha256 artifacts/handoff-20261119.zip.sha256
mkdir -p artifacts/research-v2-20261117/technical
uv run python -m scripts.review_evidence
```

原正式实验可用`rl-retest prepare/run/verify/analyze`和configs/study.json在新的输出目录复现。原种子20261123、开发20261108、bootstrap20261129固定。不得把重复复现计为新独立证据。

## 信任与解释范围

SHA256SUMS校验所有打包文件；旁置zip.sha256校验压缩包。package_checks.json核对数量、来源哈希、审阅版本及核心图文件存在；它不代替研究验收。原源码基线Git提交4672e95，工作修改与新增脚本全部在source快照中，实际版本以内容摘要为准。

95%区间逐比较而非整份报告联合保证；精确任务值不附抽样区间。大规模真最优未知，耗时只是单worker观测值。成员实际贡献继续据实记录。PPT文件由接手成员制作，本包提供完整正文、图注及讲解依据。
''')
    files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(sha(f)+'  '+f.relative_to(OUT).as_posix()+'\n' for f in files))
    for line in (OUT/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(OUT/name)==digest
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'verified':True,'files':len(files)+1}))
if __name__=='__main__':main()
