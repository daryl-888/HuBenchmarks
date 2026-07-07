#!/usr/bin/env python3
"""Gather per-category breakdowns from completed runs."""
import json, os, glob

categories = {}
for d in sorted(glob.glob('/project/rhu/dpalfaro/results/*run1/')):
    name = os.path.basename(d.rstrip('/'))
    if name in ['mdp3_run1', 'dycoke_run1', 'baseline_run1', 'fastvid_run1', 'dyto_run1']:
        continue
    rfile = os.path.join(d, 'results.jsonl')
    if not os.path.exists(rfile):
        subdirs = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
        for sd in subdirs:
            rfile = os.path.join(sd, 'results.jsonl')
            if os.path.exists(rfile): break
    if not os.path.exists(rfile): continue
    cats = {}
    with open(rfile) as f:
        for line in f:
            r = json.loads(line)
            qt = r.get('question_type', 'Unknown')
            sc = r.get('correct')
            if qt not in cats: cats[qt] = {'c': 0, 't': 0}
            if sc is not None:
                cats[qt]['t'] += 1
                if sc: cats[qt]['c'] += 1
    categories[name] = cats

all_cats = sorted(set(c for cats in categories.values() for c in cats))
print('Model', end='')
for cat in all_cats:
    print(f',{cat}', end='')
print()
for name, cats in sorted(categories.items()):
    print(name, end='')
    for cat in all_cats:
        if cat in cats:
            d = cats[cat]
            acc = d['c'] / d['t'] if d['t'] > 0 else 0
            print(f',{acc:.4f}', end='')
        else:
            print(',-', end='')
    print()