"""Read existing compact train trajectories and quantify local credit semantics."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'analysis/evaluation-trends';out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in (11,23,37):
        path=ROOT/f'results/training_credit/asct-uniform-seed{seed}-train.jsonl.gz'
        counts=dict(trajectories=0,steps=0,multi_action_steps=0,chosen_sign_changes=0,
                    multi_chosen_sign_changes=0,positive_terminal_steps=0,positive_terminal_negative_credit=0)
        iterations={};qids=set();logical=set()
        with gzip.open(path,'rt') as f:
            for line in f:
                row=json.loads(line);counts['trajectories']+=1;qids.add(row['qid'])
                iterations[row['iteration']]=iterations.get(row['iteration'],0)+1
                for step in row['steps']:
                    p=np.array(step['probabilities']); q=np.array(step['q_utility']); a=step['labels'].index(step['chosen_label'])
                    assert len(p)==len(q) and abs(sum(p)-1)<1e-5
                    assert abs(q[a]-p@q-step['advantage'])<1e-5
                    logical.add(step['search']['logical_evaluations'])
                    adv=q[a]-p@q; uniform=q[a]-q.mean()
                    changed=abs(adv)>1e-8 and abs(uniform)>1e-8 and np.sign(adv)!=np.sign(uniform)
                    counts['steps']+=1;counts['chosen_sign_changes']+=int(changed)
                    if len(q)>=3:
                        counts['multi_action_steps']+=1;counts['multi_chosen_sign_changes']+=int(changed)
                    if row['terminal']['utility']>0:
                        counts['positive_terminal_steps']+=1
                        counts['positive_terminal_negative_credit']+=int(adv < -1e-8)
        rows.append(dict(seed=seed,**counts,unique_train_qids=len(qids),logical_evaluations=sorted(logical),iterations=iterations))
    (out/'existing_train_audit.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
