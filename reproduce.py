"""Portable offline numerical reconstruction; no training or inference."""
from pathlib import Path
import hashlib,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
for r in manifest['files']:
    p=ROOT/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],r['path']
out=ROOT/'_checks';out.mkdir(exist_ok=True)
commands=['tools/verify_results.py','tools/analyze_new_runs.py','tools/audit_training_credit.py','paper_data/build_probe_assets.py','paper_data/build_v16_assets.py','paper_data/build_split_figures.py','paper_data/build_endpoint_figures.py','paper_data/build_difficulty_figure.py','paper_data/build_budget_figures.py','paper_data/build_token_composition.py','paper_data/build_emphasis_figures.py','paper_data/build_figure7.py']
env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'matplotlib'))
for n in commands:
    result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/n)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/(Path(n).stem+'.log')).write_bytes(result.stdout)
    if result.returncode:
        print(result.stdout.decode('utf-8',errors='replace'));sys.exit(result.returncode)
    print('PASS',n,flush=True)
for name in ['difficulty_by_seed.csv','difficulty_summary.csv']:
    assert (ROOT/'analysis/learning-diagnostics'/name).read_bytes() == (ROOT/'paper_data/data'/name).read_bytes(), name
assert (ROOT/'results/experiment-3-meta-token/search_costs.csv').read_bytes() == (ROOT/'paper_data/data/search_costs.csv').read_bytes()
summary=json.loads((ROOT/'analysis_new_controls/summary.json').read_text())
assert abs(summary['actor_cf']['test']['utility_official']['mean']-0.5974767096043987)<1e-12
assert abs(summary['at2po']['test']['utility_official']['mean']-0.5646725193311856)<1e-12
credit=json.loads((ROOT/'analysis/evaluation-trends/existing_train_audit.json').read_text())
assert sum(r['multi_action_steps'] for r in credit)==36000
assert sum(r['multi_chosen_sign_changes'] for r in credit)==3117
assert sum(r['steps'] for r in credit)==158402
(out/'result.json').write_text(json.dumps({'status':'passed','input_files':len(manifest['files']),'checks':commands,'training_rerun':False},indent=2))
print('Offline numerical reconstruction PASS. No training or inference was run.')
