"""Versioned JSON records and atomic per-job storage."""
import json
import os
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict
from ..policy import Stop, Test


class ResultRecord(TypedDict):
    schema_version: int
    job_id: str
    task_id: str
    method: str
    memory: str
    status: str
    policy: NotRequired[dict]
    expected_utility: NotRequired[float]
    expected_tests: NotRequired[float]
    success_probability: NotRequired[float]
    model_value: NotRequired[float | None]
    lower: NotRequired[float | None]
    upper: NotRequired[float | None]
    evaluations: NotRequired[int]
    error: NotRequired[str]


def policy_to_data(policy):
    if isinstance(policy, Stop):
        return {'action': 'stop', 'path': policy.path}
    return {'action': 'test', 'path': policy.path, 'failure': policy_to_data(policy.failure),
            'success': policy_to_data(policy.success)}


def policy_from_data(data):
    if data['action'] == 'stop':
        return Stop(data['path'])
    if data['action'] == 'test':
        return Test(data['path'], policy_from_data(data['failure']), policy_from_data(data['success']))
    raise ValueError('unknown policy action')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.writing-', dir=path.parent, text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def dump_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def dump_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for row in rows))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
