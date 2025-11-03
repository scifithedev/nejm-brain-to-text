#!/usr/bin/env python3
"""
Run LM+Redis and evaluator in the two-conda-env workflow.

Usage examples:
  python3 scripts/run_with_services.py --lm-path language_model/pretrained_language_models/openwebtext_1gram_lm_sil -- --eval_type test --max_trials 1
  python3 scripts/run_with_services.py --lm-path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --detach --lm-ready-timeout 120 -- --eval_type test --max_trials 5

This script is a Python wrapper around the existing helper logic. It is intentionally
editable so you can tune timeouts, env names, and args.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time


def run(cmd, check=True, capture=False, shell=False):
    if isinstance(cmd, (list, tuple)):
        proc = subprocess.run(cmd, check=check, capture_output=capture, text=True, shell=shell)
    else:
        proc = subprocess.run(cmd, check=check, capture_output=capture, text=True, shell=True)
    return proc


def write_status(status):
    os.makedirs('data/diagnostics', exist_ok=True)
    with open('data/diagnostics/service_status.json', 'w') as f:
        json.dump(status, f, indent=2)


def ensure_conda():
    try:
        run(['conda', '--version'], check=True, capture=True)
    except Exception as e:
        print('ERROR: conda not found in PATH. Install Miniconda/Anaconda or add conda to PATH.', file=sys.stderr)
        sys.exit(1)


def redis_ping_using_cli(redis_cli='redis-cli'):
    try:
        run([redis_cli, 'ping'], check=True, capture=True)
        return True
    except Exception:
        return False


def start_redis_if_needed(lm_env, redis_server='redis-server', redis_cli='redis-cli'):
    if redis_ping_using_cli(redis_cli):
        print('Redis ping OK.')
        return

    # Try system redis-server
    if shutil.which(redis_server):
        print('Starting system redis-server as daemon')
        run([redis_server, '--daemonize', 'yes'])
        time.sleep(1)
        if redis_ping_using_cli(redis_cli):
            print('Started system redis-server')
            return

    # Try starting redis-server in LM conda env
    print(f"Attempting to start redis-server inside conda env '{lm_env}'")
    try:
        run(['conda', 'run', '-n', lm_env, '--no-capture-output', 'bash', '-lc', f"{redis_server} --daemonize yes"], check=True)
        time.sleep(1)
        if run(['conda', 'run', '-n', lm_env, '--no-capture-output', 'bash', '-lc', f"redis-cli ping"], check=False, capture=True).returncode == 0:
            print('Started redis-server inside LM env')
            return
    except Exception:
        pass

    print('ERROR: unable to start redis-server. Please install Redis or start it manually.', file=sys.stderr)
    sys.exit(1)


def lm_process_is_running(lm_script):
    try:
        proc = subprocess.run(['pgrep', '-f', lm_script], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def start_lm(lm_env, lm_script, lm_path, redis_host, redis_port):
    # Start LM server in background via conda run
    cmd = (
        f"conda run -n {shlex.quote(lm_env)} --no-capture-output bash -lc "
        f"\"nohup python {shlex.quote(lm_script)} --lm_path {shlex.quote(lm_path)} --redis_ip {shlex.quote(redis_host)} --redis_port {int(redis_port)} > /tmp/lm_server.log 2>&1 & echo $! > /tmp/lm_server.pid\""
    )
    print('Starting LM server...')
    run(cmd, check=True, shell=True)
    time.sleep(1)
    if lm_process_is_running(lm_script):
        pid = None
        try:
            with open('/tmp/lm_server.pid') as f:
                pid = f.read().strip()
        except Exception:
            pass
        print(f'LM server started (pid {pid}). Logs: /tmp/lm_server.log')
    else:
        print('ERROR: failed to start LM server; check /tmp/lm_server.log', file=sys.stderr)
        sys.exit(1)


def kill_lm(lm_script):
    try:
        subprocess.run(['pkill', '-f', lm_script])
        time.sleep(1)
    except Exception:
        pass


def wait_for_lm_ready(timeout=60):
    waited = 0
    ready = False
    print(f'Waiting up to {timeout}s for LM server to initialize (checking /tmp/lm_server.log)...')
    while waited < timeout:
        if os.path.exists('/tmp/lm_server.log'):
            try:
                with open('/tmp/lm_server.log') as f:
                    tail = f.read()
                if any(s in tail for s in ('Language model successfully initialized', 'Connected to redis', 'LM ready', 'listening')):
                    print('LM server signaled ready.')
                    ready = True
                    break
            except Exception:
                pass
        time.sleep(1)
        waited += 1
    if not ready:
        print(f'ERROR: LM server did not signal ready within {timeout}s. Check /tmp/lm_server.log', file=sys.stderr)
        return False
    return True


def run_evaluator(eval_env, eval_args, detach=False):
    if detach:
        if not eval_args:
            eval_args = '--eval_type test --max_trials 5'
        cmd = (
            f"conda run -n {shlex.quote(eval_env)} --no-capture-output bash -lc "
            f"\"nohup env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py {eval_args} > /tmp/evaluator.log 2>&1 & echo $! > /tmp/evaluator.pid\""
        )
        print('Starting evaluator in detached mode...')
        run(cmd, check=True, shell=True)
        # Wait briefly for the pid file to be written by the background shell
        pid = None
        for _ in range(10):
            try:
                if os.path.exists('/tmp/evaluator.pid') and os.path.getsize('/tmp/evaluator.pid') > 0:
                    with open('/tmp/evaluator.pid') as f:
                        pid = f.read().strip()
                        break
            except Exception:
                pass
            time.sleep(0.2)
        print(f'Evaluator started (pid {pid}). Log: /tmp/evaluator.log')

        # Wait for the detached evaluator to exit so the caller doesn't have to run
        # separate status-checking scripts. We poll the process and then read the
        # status JSON for the evaluator exit code.
        if pid is None:
            print('Warning: evaluator pid not found; returning.')
            return 0

        try:
            pid_int = int(pid)
        except Exception:
            # Fall back: try to detect the evaluator process via pgrep
            try:
                p = subprocess.run(['pgrep', '-f', 'model_training/evaluate_model.py'], capture_output=True, text=True)
                candidates = [int(x) for x in p.stdout.split() if x.strip()]
                if candidates:
                    pid_int = candidates[-1]
                    print(f'Falling back to detected evaluator pid {pid_int}')
                else:
                    print('Warning: invalid pid in /tmp/evaluator.pid and no pgrep candidates; returning.')
                    return 0
            except Exception:
                print('Warning: invalid pid in /tmp/evaluator.pid and pgrep failed; returning.')
                return 0

        # Poll until process exits
        print(f'Waiting for evaluator (pid {pid_int}) to finish...')
        while True:
            proc = subprocess.run(['ps', '-p', str(pid_int)], capture_output=True)
            if proc.returncode != 0:
                break
            time.sleep(1)

        # Read status file for last_return_code if present
        retcode = 0
        try:
            with open('data/diagnostics/service_status.json') as f:
                st = json.load(f)
            retcode = int(st.get('last_return_code', 0))
        except Exception:
            retcode = 0

        print(f'Evaluator (pid {pid_int}) finished with return code {retcode}.')
        print('Tail of evaluator log:')
        try:
            print('\n'.join(subprocess.run(['tail', '-n', '100', '/tmp/evaluator.log'], capture_output=True, text=True).stdout.splitlines()))
        except Exception:
            pass

        return retcode
    else:
        if not eval_args:
            args = ['--eval_type', 'test', '--max_trials', '1']
        else:
            args = shlex.split(eval_args)
        cmd = ['conda', 'run', '-n', eval_env, '--no-capture-output', 'env', 'B2TXT_SKIP_ENV_CHECK=1', 'python', 'model_training/evaluate_model.py'] + args
        print('Running evaluator in foreground (this will block until completion)...')
        return subprocess.call(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-path', type=str, default='language_model/pretrained_language_models/openwebtext_1gram_lm_sil')
    parser.add_argument('--lm-env', type=str, default='b2txt25_lm')
    parser.add_argument('--eval-env', type=str, default='b2txt25')
    parser.add_argument('--lm-ready-timeout', type=int, default=60)
    parser.add_argument('--redis-host', type=str, default='127.0.0.1')
    parser.add_argument('--redis-port', type=int, default=6379)
    parser.add_argument('--detach', action='store_true', help='Run evaluator detached')
    parser.add_argument('--force-restart-lm', action='store_true', help='Always restart LM even if running')
    parser.add_argument('--lm-script', type=str, default='language_model/language-model-standalone.py')
    parser.add_argument('remaining', nargs=argparse.REMAINDER, help='Evaluator args after --')
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_conda()

    lm_path = args.lm_path
    lm_env = args.lm_env
    eval_env = args.eval_env
    lm_script = args.lm_script
    redis_host = args.redis_host
    redis_port = args.redis_port
    timeout = args.lm_ready_timeout

    eval_argstr = ''
    if args.remaining:
        # remaining starts with '--' if delimiter used; strip it
        rem = args.remaining
        if rem and rem[0] == '--':
            rem = rem[1:]
        eval_argstr = ' '.join(shlex.quote(x) for x in rem)

    # Ensure Redis is up (basic)
    try:
        import shutil
    except Exception:
        pass

    # Basic check: try redis-cli ping
    if not redis_ping_using_cli('redis-cli'):
        # try starting redis via conda/LM env
        start_redis_if_needed(lm_env)

    # If LM is running and force-restart is requested or configured redis does not match, restart
    if lm_process_is_running(lm_script):
        if args.force_restart_lm:
            print('Force restart requested: killing existing LM process')
            kill_lm(lm_script)
        else:
            # inspect /tmp/lm_server.log for configured ip
            configured = None
            if os.path.exists('/tmp/lm_server.log'):
                try:
                    with open('/tmp/lm_server.log') as f:
                        lines = f.read()
                    # find last "Attempting to connect to redis at X:Y"
                    import re
                    m = re.findall(r'Attempting to connect to redis at ([0-9\.]+):([0-9]+)', lines)
                    if m:
                        configured = m[-1]
                except Exception:
                    configured = None
            if configured and (configured[0] != redis_host or int(configured[1]) != int(redis_port)):
                print(f"Existing LM points to {configured[0]}:{configured[1]} which does not match {redis_host}:{redis_port}; restarting LM.")
                kill_lm(lm_script)

    # Start LM
    if not lm_process_is_running(lm_script):
        start_lm(lm_env, lm_script, lm_path, redis_host, redis_port)

    status = {
        'lm_path': lm_path,
        'redis_host': redis_host,
        'redis_port': redis_port,
        'lm_pid_file': '/tmp/lm_server.pid',
        'lm_log': '/tmp/lm_server.log',
        'evaluator_pid_file': '/tmp/evaluator.pid',
        'evaluator_log': '/tmp/evaluator.log',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    write_status(status)

    if not wait_for_lm_ready(timeout):
        write_status(status)
        sys.exit(2)

    # Run evaluator
    ret = run_evaluator(eval_env, eval_argstr, detach=args.detach)
    # update status with exit/ret
    status['last_return_code'] = int(ret)
    write_status(status)
    sys.exit(int(ret))


if __name__ == '__main__':
    main()
