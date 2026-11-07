"""Command line entry point; imports work only when commands execute."""
import argparse
import json
import os


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rl-retest')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--suite', required=True, choices=['dev','main','large','stress','pruning'])
    prepare.add_argument('--out', required=True)
    run = commands.add_parser('run')
    run.add_argument('--run-dir', required=True)
    run.add_argument('--workers', type=int, default=1)
    for name in ('verify','analyze'):
        command = commands.add_parser(name)
        command.add_argument('--run-dir', required=True)
    args = parser.parse_args(argv)
    # Set thread limits before importing NumPy through the experiment modules.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if args.command == 'prepare':
        from .experiments.runner import prepare as execute
        manifest = execute(args.config,args.suite,args.out)
        print(f"Prepared {manifest['task_count']} tasks and {manifest['job_count']} jobs")
    elif args.command == 'run':
        from .experiments.runner import run as execute
        print(f"Executed {execute(args.run_dir,args.workers)} pending jobs")
    elif args.command == 'verify':
        from .experiments.runner import verify as execute
        result = execute(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['valid'] else 1
    else:
        from .analysis import analyze
        print(json.dumps(analyze(args.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
