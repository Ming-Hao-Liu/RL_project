"""Check portable research entrypoints and current documentation navigation."""
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    'analyze_study',
    'analyze_history',
    'analyze_sequential',
    'analyze_search',
    'diagnose_tolerance',
    'review_evidence',
    'run_research',
    'analyze_mechanisms',
    'analyze_calibration',
    'time_initialization',
    'time_deep',
    'verify_research',
)


@pytest.fixture
def script_environment(tmp_path):
    """Run outside the repository while keeping every cache inside the project."""
    working = tmp_path / 'working'
    working.mkdir()
    cache = tmp_path / 'cache'
    cache.mkdir()
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR=str(cache / 'matplotlib'),
        XDG_CACHE_HOME=str(cache),
    )
    return working, env


def test_research_scripts_import_without_experiment_io(script_environment):
    working, env = script_environment
    # Imported libraries may read their own resources and populate the cache.
    # Research data and output directories must remain untouched by imports.
    code = '''
import importlib
from pathlib import Path
import sys

roots = [Path(sys.argv[1]), Path.cwd()]
restricted = [root / name for root in roots
              for name in ('data', 'results', 'artifacts')]

def audit(event, args):
    if event not in ('open', 'os.mkdir') or not args:
        return
    path = args[0]
    if not isinstance(path, (str, bytes)):
        return
    path = Path(path.decode() if isinstance(path, bytes) else path).resolve()
    if any(path.is_relative_to(root) for root in restricted):
        raise AssertionError(f'experiment I/O during import: {path}')

sys.addaudithook(audit)
for name in sys.argv[2:]:
    importlib.import_module('scripts.' + name)
'''
    subprocess.run(
        [sys.executable, '-c', code, str(ROOT), *SCRIPTS],
        cwd=working, env=env, check=True, capture_output=True, text=True,
    )
    assert not list(working.iterdir())


@pytest.mark.parametrize('module', SCRIPTS)
def test_research_script_help_without_local_data(module, script_environment):
    working, env = script_environment
    result = subprocess.run(
        [sys.executable, '-m', f'scripts.{module}', '--help'],
        cwd=working, env=env, check=True, capture_output=True, text=True,
    )
    assert '--source' in result.stdout
    assert '--out' in result.stdout
    assert not list(working.iterdir())


def test_current_markdown_links_resolve():
    documents = [ROOT / 'README.md', ROOT / 'data/README.md', ROOT / 'results/README.md']
    documents.extend(sorted((ROOT / 'docs').glob('*.md')))
    errors = []
    for document in documents:
        assert document.is_file(), document
        content = re.sub(r'```.*?```', '', document.read_text(), flags=re.DOTALL)
        for raw in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)', content):
            target = raw.strip()
            target = target[1:target.index('>')] if target.startswith('<') else target.split()[0]
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            destination = (document.parent / unquote(parsed.path)).resolve()
            # A fresh clone has only the two tracked directory READMEs.
            # Once formal evidence is installed, check all its linked outputs.
            evidence_markers = {
                'data': ROOT / 'data/main/manifest.json',
                'results': ROOT / 'results/study/provenance.json',
            }
            optional = any(
                destination.is_relative_to(ROOT / folder)
                and destination != ROOT / folder / 'README.md'
                and not marker.exists()
                for folder, marker in evidence_markers.items()
            )
            if optional:
                continue
            if not destination.exists():
                errors.append(f'{document.relative_to(ROOT)}: {target}')
    assert not errors, '\n'.join(errors)
