"""Protect the dependency boundaries that make independent evaluation possible."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1] / 'src' / 'rl_retest'


def imports(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield node.module or ''
        elif isinstance(node, ast.Import):
            yield from (item.name for item in node.names)


def test_dependency_directions():
    for name in ('model.py','policy.py'):
        assert not any(x.startswith(('rl_retest.planners','planners','experiments','evaluation','analysis')) for x in imports(PACKAGE / name))
    for path in (PACKAGE / 'planners').glob('*.py'):
        assert not any(x.startswith(('rl_retest.evaluation','evaluation','experiments','analysis')) for x in imports(path))
    for name in ('evaluation.py','analysis.py'):
        assert not any(x.startswith(('rl_retest.planners','planners')) for x in imports(PACKAGE / name))
    oracle = Path(__file__).parent / 'oracle.py'
    assert not any(x.startswith(('rl_retest','model','planners','bounds')) for x in imports(oracle))


def test_project_import_graph_is_acyclic():
    paths = list(PACKAGE.rglob('*.py'))
    def module_name(path):
        parts = list(path.relative_to(PACKAGE).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        return '.'.join(['rl_retest', *parts])
    modules = {module_name(path): path for path in paths}
    graph = {name:set() for name in modules}
    for name,path in modules.items():
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                target = (importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                          if node.level else node.module or '')
                if target in modules and target != name:
                    graph[name].add(target)
            elif isinstance(node, ast.Import):
                graph[name].update(alias.name for alias in node.names if alias.name in modules and alias.name != name)
    active, done = set(), set()
    def visit(name):
        assert name not in active, f'cyclic import involving {name}'
        if name in done:
            return
        active.add(name)
        for target in graph[name]:
            visit(target)
        active.remove(name)
        done.add(name)
    for module in graph:
        visit(module)


def test_import_has_no_working_directory_side_effects(tmp_path):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    before = set(tmp_path.iterdir())
    subprocess.run([sys.executable, '-c', 'import rl_retest.model, rl_retest.policy, rl_retest.evaluation, rl_retest.planners.bounded'],
                   cwd=tmp_path, env=env, check=True)
    assert set(tmp_path.iterdir()) == before
