"""Run each implementation's tests in a separate process to isolate snapshots."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'_checks';OUT.mkdir(exist_ok=True)
for name in ['main','actor_rollout','at2po']:
    source=ROOT/'code'/name
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=os.pathsep.join(map(str,[source/'src',source/'packages/agent-uct'])))
    p=subprocess.run([sys.executable,'-m','pytest',str(source/'tests'),'-q','-ra','-p','no:cacheprovider'],cwd=source,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/f'tests_{name}.log').write_bytes(p.stdout)
    print(name,p.stdout.decode('utf-8',errors='replace'),flush=True)
    if p.returncode:sys.exit(p.returncode)
print('Test suites completed; inspect skipped dependency-dependent cases in _checks/.')
