"""Generate traceable Markdown tables for the two final experiment reports."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SIZE = {name:index for index,name in enumerate(("tiny","small","medium","large"))}
ALG = {name:index for index,name in enumerate(("reinforce","a2c","ppo"))}

def read(folder, name):
    return json.loads((folder / (name + ".json")).read_text())["rows"]

def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows) + "\n"

def number(value, digits=2):
    return "?" if value is None else f"{value:.{digits}f}"

def interval(stats, digits=2):
    return f"[{stats['confidence_interval_low']:.{digits}f}, {stats['confidence_interval_high']:.{digits}f}]"

def group(rows, cell):
    return [r for r in rows if all(r[k] == cell[k] for k in ("algorithm","actor_name","observation_type") if k in cell)]

def condition(cell):
    return f"{cell['algorithm']}/{cell['actor_name']}" if "algorithm" in cell else cell['observation_type']

def average(rows, key):
    return float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None

for experiment, folder_name in ((1,"experiment_1"),(2,"experiment_2_revised")):
    folder=ROOT / "results/analysis/reported_experiments" / folder_name
    cells=read(folder,"cell_summaries")
    summaries=read(folder,"run_summaries")
    controls=read(folder,"root_controls")
    pairs=read(folder,"paired_summaries")
    curves=read(folder,"common_budget_outcomes")
    updates=read(folder,"optimization_diagnostics")
    cells.sort(key=lambda r:(ALG.get(r.get('algorithm'),0),SIZE.get(r.get('actor_name'),0),r.get('observation_type','')))
    text=f"# Experiment {experiment}: generated evidence tables\n\n"
    text+="## Final outcomes\n\n"+table(["Condition","Completed laps","Return mean ? SD","Return 95% interval","Progress","Lap time (s)","Threshold roots"],[
      [condition(c),f"{c['completed_lap_count']}/{c['completed_lap_denominator']}",f"{c['final_mean_return']['mean']:.2f} ? {c['final_mean_return']['sample_standard_deviation']:.2f}",interval(c['final_mean_return']),number(c['final_mean_progress']['mean'],3),number(None if c['completed_lap_time'] is None else c['completed_lap_time']['mean']),f"{c['converged_root_count']}/{c['root_count']}"] for c in cells])
    text+="\n## Threshold and stability\n\nTimes/episodes below average only observed attainments; the denominator reports censoring. Late completion is the mean validation/fixed-track completion over the final 20% of the budget.\n\n"+table(["Condition","Attained","First interactions range","Mean episodes to first","Mean first training time (s)","Mean confirmation time (s)","Late completion"],[
      [condition(c),f"{c['converged_root_count']}/{c['root_count']}",("?" if not (v:=[r['convergence_interactions'] for r in group(summaries,c) if r['converged']]) else f"{min(v)/1000:g}k?{max(v)/1000:g}k"),number(average(group(summaries,c),'episodes_to_convergence'),0),number(average(group(summaries,c),'convergence_duration'),1),number(average(group(summaries,c),'confirmation_duration'),1),number(c['late_completion_rate']['mean'],3)] for c in cells])
    if experiment==1:
      text+="\n## Common-budget return and completion\n\nEach entry is mean return (completed roots / 5) at the exact evaluation boundary.\n\n"+table(["Condition","250k","500k","750k","1M","2M"],[
        [condition(c)]+[f"{np.mean([r['mean_return'] for r in group(curves,c) if r['training_interactions']==b]):.1f} ({sum(r['completion_rate']==1 for r in group(curves,c) if r['training_interactions']==b)}/5)" for b in (250000,500000,750000,1000000,2000000)] for c in cells])
    text+="\n## Paired return contrasts\n\nPositive favours the first named condition.\n\n"+table(["Fixed condition","Contrast","Mean difference","95% interval","Mean return AUC difference"],[
      [p['algorithm'] or p['actor_name'],p['contrast'],number(p['final_mean_return']['mean']),interval(p['final_mean_return']),number(p['return_auc']['mean'])] for p in pairs])
    text+="\n## Learned controls\n\nMeans over roots after equal circuit weighting within each root. Requested steering reversal rate ignores magnitudes ?0.05.\n\n"+table(["Condition","Observed coverage","Speed mean","Braking fraction","Throttle q10 / median / q90","Throttle SD","Mean steering change","Steering reversals / s"],[
      [condition(c),number(average(group(controls,c),'coverage'),3),number(average(group(controls,c),'mean_speed')),number(average(group(controls,c),'braking_fraction'),3),' / '.join(number(average(group(controls,c),k)) for k in ('throttle_q10','throttle_median','throttle_q90')),number(average(group(controls,c),'throttle_standard_deviation')),number(average(group(controls,c),'mean_steering_change'),3),number(average(group(controls,c),'steering_reversals_per_time'))] for c in cells])
    text+="\n## Resources\n\nMean full-budget durations; these do not measure time to threshold.\n\n"+table(["Condition","Collection (min)","Optimization (min)","Evaluation (min)","End-to-end (min)","Collection step/s"],[
      [condition(c)]+[number(average(group(summaries,c),k)/60) for k in ('collection_duration','optimization_duration','evaluation_duration','end_to_end_duration')]+[number(average(group(summaries,c),'collection_throughput'),0)] for c in cells])
    late_updates=[]
    for r in summaries:
      u=sorted([x for x in updates if x['run_id']==r['run_id']],key=lambda x:x['update_index'])
      u=u[int(0.9*len(u)):]
      late_updates.append({**r,**{k:average(u,k) for k in ('explained_variance','actor_gradient_norm','approximate_kl','clip_fraction','log_standard_deviation_0','log_standard_deviation_1')}})
    text+="\n## Optimization diagnostics\n\nEach root is averaged over its final tenth of updates, then roots are averaged equally.\n\n"+table(["Condition","Explained variance","Actor gradient norm","Approx. KL","Clip fraction","log ? throttle / steer"],[
      [condition(c)]+[number(average(group(late_updates,c),k),4 if k=='approximate_kl' else 3) for k in ('explained_variance','actor_gradient_norm','approximate_kl','clip_fraction')]+[' / '.join(number(average(group(late_updates,c),k),3) for k in ('log_standard_deviation_0','log_standard_deviation_1'))] for c in cells])
    if experiment==2:
      text+="\n## Paired test completion\n\n"+table(["Contrast","Difference","95% interval"],[[p['contrast'],number(p['final_completion_rate']['mean'],3),interval(p['final_completion_rate'],3)] for p in pairs])
      text+="\n"+table(["Root","Frenet completion","LiDAR completion","Difference","Return difference"],[
       [root,number((f:=next(r for r in summaries if r['root_identity']==root and r['observation_type']=='frenet'))['final_completion_rate'],3),number((l:=next(r for r in summaries if r['root_identity']==root and r['observation_type']=='lidar'))['final_completion_rate'],3),number(f['final_completion_rate']-l['final_completion_rate'],3),number(f['final_mean_return']-l['final_mean_return'])] for root in range(10)])
      splits=read(folder,'final_split_summaries')
      text+="\n## Generalization splits\n\n"+table(["Observation","Split","Mean completion","Mean progress","Mean return"],[
       [obs,split,number(average((s:=[r for r in splits if r['observation_type']==obs and r['circuit_split']==split]),'completion_rate'),3),number(average(s,'mean_progress'),3),number(average(s,'mean_return'))] for obs in ('frenet','lidar') for split in ('training_reference','validation','test')])
    (ROOT/'results/analysis/provenance'/f'experiment_{experiment}_tables.md').write_text(text,encoding='utf-8')
    print(experiment,len(summaries),'runs; evidence tables generated')
