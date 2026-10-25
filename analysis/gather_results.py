#!/usr/bin/env python3
"""Gather all completed model results from Carya and print a clean table."""
import json, os, glob

results = {}
for d in sorted(glob.glob('/project/rhu/dpalfaro/results/*run1/')):
    name = os.path.basename(d.rstrip('/'))
    data = {}
    summary_file = os.path.join(d, 'summary.json')
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            data = json.load(f)
        results[name] = {
            'accuracy': data.get('accuracy', '?'),
            'correct': data.get('correct', '?'),
            'total_scoreable': data.get('total_scoreable', '?'),
            'total_na_skipped': data.get('total_na_skipped', '?'),
            'model': data.get('model', '?'),
        }
        for k in data:
            if k.endswith('_params'):
                results[name]['params'] = data[k]
    else:
        rfile = os.path.join(d, 'results.jsonl')
        if os.path.exists(rfile):
            c = t = n = 0
            model = '?'
            with open(rfile) as f:
                for line in f:
                    r = json.loads(line)
                    sc = r.get('correct')
                    if sc is not None:
                        t += 1
                        if sc: c += 1
                    else:
                        n += 1
                    if 'model' in r: model = r['model']
            acc = c / t if t > 0 else 0
            results[name] = {
                'accuracy': acc,
                'correct': c,
                'total_scoreable': t,
                'total_na_skipped': n,
                'model': model,
            }
        else:
            subdirs = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
            for sd in subdirs:
                rfile = os.path.join(sd, 'results.jsonl')
                if os.path.exists(rfile):
                    c = t = n = 0
                    model = os.path.basename(sd)
                    with open(rfile) as f:
                        for line in f:
                            r = json.loads(line)
                            sc = r.get('correct')
                            if sc is not None:
                                t += 1
                                if sc: c += 1
                            else:
                                n += 1
                    acc = c / t if t > 0 else 0
                    results[name] = {
                        'accuracy': acc,
                        'correct': c,
                        'total_scoreable': t,
                        'total_na_skipped': n,
                        'model': model,
                    }
                    break

# Print table
print(f"{'Model':<25} {'Accuracy':>10} {'Correct':>8} {'Scoreable':>10} {'NA':>8} {'Backbone':<30}")
print('-' * 95)
for name, data in sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
    acc = data.get('accuracy', '?')
    acc_str = f'{acc:.4f}' if isinstance(acc, float) else str(acc)
    model = data.get('model', '?')
    if '/' in model:
        model = model.split('/')[-1]
    print(f'{name:<25} {acc_str:>10} {str(data.get("correct","?")):>8} {str(data.get("total_scoreable","?")):>10} {str(data.get("total_na_skipped","?")):>8} {model:<30}')