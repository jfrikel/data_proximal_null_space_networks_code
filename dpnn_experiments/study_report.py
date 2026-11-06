# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only complete-case accounting, paired analysis and portable SVG/HTML output."""
import csv
import html
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from dpnn_experiments.main_integrity import SealedArchive, read, implementation
from dpnn_experiments.training_archive import file_reference, verify_reference
from dpnn_experiments.provenance import write_json, utc_now
from dpnn_experiments.paired_statistics import describe, paired_effect
from dpnn_experiments import study_plan as plan


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def training_segments(root,job):
    """Original + continuation attempts; trim work discarded by later resumes."""
    paths=[];visiting=set()
    def parent_chain(run):
        run=Path(run).resolve()
        if run in visiting:raise ValueError('Cyclic training lineage')
        visiting.add(run)
        c=read(run/'manifest.json')['configuration'];ref=c['execution']['resume']
        if ref:parent_chain(Path(ref['path']).parent.parent)
        if run not in paths:paths.append(run)
        visiting.remove(run)
    transition=root/'transitions'/(job['id']+'.json')
    if transition.exists():
        t=read(transition);parent_chain(Path(verify_reference(t['parent_manifest'])).parent)
    for p in sorted((root/'jobs'/job['id']).glob('attempt_*/runs/*/completion.json')):
        parent_chain(p.parent)
    limits={}
    for run in paths:
        manifest=read(run/'manifest.json');ref=manifest['configuration']['execution']['resume']
        if not ref:continue
        restored=[r for r in jsonlines(run/'events.jsonl') if r.get('event')=='resume_restored']
        if not restored:continue
        progress=restored[-1]['position'];parent=Path(ref['path']).parent.parent
        limits[parent]=(progress['step'],progress['phase'])
    return [(run,limits.get(run)) for run in paths]


def table(path,rows):
    if not rows:path.write_text('');return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def chart(path,title,series,xlabel,ylabel):
    """Each series: label, [(x,y)], optional interval [lower,upper] per point."""
    colors=['#164e8b','#bb3d20','#237a40','#8c47ad','#c18c00','#444444']
    points=[(x,y) for _,xy,*_ in series for x,y in xy if np.isfinite(x) and np.isfinite(y)]
    if not points:points=[(0,0),(1,1)]
    xmin,xmax=min(x for x,y in points),max(x for x,y in points)
    ys=[y for x,y in points]+[v for s in series if len(s)>2 for ci in s[2] if ci for v in ci]
    ymin,ymax=min(ys),max(ys)
    if xmin==xmax:xmin-=.5;xmax+=.5
    if ymin==ymax:ymin-=.5;ymax+=.5
    pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad
    X=lambda x:85+710*(x-xmin)/(xmax-xmin)
    Y=lambda y:420-350*(y-ymin)/(ymax-ymin)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500" viewBox="0 0 1100 500">',
        '<rect width="1100" height="500" fill="white"/>',f'<text x="30" y="25" font-size="18">{html.escape(title)}</text>',
        '<path d="M85 60 V420 H795" fill="none" stroke="black"/>']
    for k in range(6):
        x=xmin+(xmax-xmin)*k/5;y=ymin+(ymax-ymin)*k/5
        out.extend([f'<text x="{X(x):.1f}" y="440" font-size="10">{x:.3g}</text>',f'<text x="5" y="{Y(y):.1f}" font-size="10">{y:.4g}</text>'])
    for k,s in enumerate(series):
        label,xy=s[:2];color=colors[k%len(colors)]
        if len(xy)<200:
            out.append('<polyline fill="none" stroke="'+color+'" points="'+' '.join(f'{X(x):.1f},{Y(y):.1f}' for x,y in xy if np.isfinite(x) and np.isfinite(y))+'"/>')
        for i,(x,y) in enumerate(xy):
            if not np.isfinite(x) or not np.isfinite(y):continue
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2" fill="{color}"/>')
            if len(s)>2 and s[2][i]:
                lo,hi=s[2][i];out.append(f'<path d="M{X(x):.1f} {Y(lo):.1f} V{Y(hi):.1f}" stroke="{color}"/>')
        out.append(f'<text x="810" y="{55+18*k}" fill="{color}" font-size="11">{html.escape(str(label))}</text>')
    out.extend([f'<text x="320" y="475">{html.escape(xlabel)}</text>',f'<text x="85" y="52">{html.escape(ylabel)}</text>','</svg>'])
    path.write_text('\n'.join(out))


def build(root,phase):
    root=Path(root);plan.verify_definition(root)
    profile=read(root/'profile.json');analysis=read(root/'analysis_plan.json');ids=read(root/'split_ids.json')[phase]
    jobs=read(root/f'{phase}_jobs.json');rows=[];ledger=[];attempts=[];refs=[];issues=[];devices=set();resumed=set()
    # Check both the expected logical results and every retained failed/interrupted attempt.
    for job in read(root/'validation_jobs.json')+(jobs if phase=='test' else []):
        folder=root/'jobs'/job['id']
        for dispatch in sorted(folder.glob('attempt_*/dispatch.json')):
            d=read(dispatch);devices.add(d['gpu_uuid']);refs.append(file_reference(dispatch))
            config=read(verify_reference(d['configuration']))
            if job['kind']=='train' and config['execution']['resume']:resumed.add(job['id'])
            markers=list(dispatch.parent.glob('runs/*/completion.json'))
            if not markers:attempts.append({'job':job['id'],'attempt':str(dispatch.parent),'status':'unsealed_or_not_started'})
            for marker in markers:
                attempts.append({'job':job['id'],'attempt':str(dispatch.parent),'completion':file_reference(marker),
                    'status':read(marker)['status'],'reason':read(marker).get('reason'),'gpu_uuid':d['gpu_uuid']})
    for job in jobs:
        record={'job':job, 'expected_objects':len(ids) if job['kind']=='evaluate' else None}
        folder=root/'jobs'/job['id'];result_path=folder/'result.json';archive=None;successful=False
        if result_path.exists():
            result=plan.job_result(root,job['id']);archive=SealedArchive(result['run']);successful=True
            refs.append(file_reference(result_path));record.update(status='completed',result=file_reference(result_path),run=result['run'],reused='reused_for' in result)
        else:
            candidates=sorted(folder.glob('attempt_*/runs/*/completion.json'))
            record['status']='missing'
            if candidates:
                marker=candidates[-1];record.update(status=read(marker)['status'],run=str(marker.parent))
                if record['status'] in ('completed','failed','interrupted'):
                    archive=SealedArchive(marker.parent,statuses=('completed','failed','interrupted'))
            issues.append('Missing completed logical job: '+job['id'])
        if archive and job['kind']=='evaluate':
            actual=archive.configuration
            from dpnn_experiments.study_queue import config_for
            try:
                if plan.protocol(actual)!=plan.protocol(config_for(root,job)):raise ValueError('Evaluation protocol differs')
                metric_path=archive.member('metrics.jsonl');metrics=jsonlines(metric_path);refs.append(file_reference(metric_path))
                seen=set()
                for row in metrics:
                    if row['id'] in seen or row['id'] not in ids:raise ValueError('Duplicate/unexpected object '+row['id'])
                    if row['method']!=job['method'] or row['start_kind']!=job['start']:raise ValueError('Method/start mismatch')
                    seen.add(row['id'])
                    rows.append({**row,'job_id':job['id'],'seed_index':job['seed_index'],'selection':job['selection'],
                        'completed_job':successful,'evaluation_manifest':file_reference(archive.root/'manifest.json')})
                missing=sorted(set(ids)-seen);record.update(observed_objects=len(seen),missing_ids=missing)
                if missing:issues.append('Missing objects: '+job['id'])
                if any(r.get('constraint_violation') is True or r.get('error_decomposition_passed') is not True or r.get('nsn_leakage_identity_passed') is False for r in metrics):issues.append('Budget/decomposition check failed: '+job['id'])
            except (ValueError,KeyError,FileNotFoundError) as error:
                record['integrity_error']=str(error);issues.append(job['id']+': '+str(error))
                for row in rows:
                    if row['job_id']==job['id']:row['completed_job']=False
        ledger.append(record)
    # Each BASE row appears once; it is referenced for each paired comparison.
    groups={}
    for row in rows:groups.setdefault((row['start_kind'],row['selection'],row['seed_index'],row['method']),[]).append(row)
    numeric=sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}-{'seed_index'})
    summaries=[]
    for key,group in groups.items():
        start,selection,seed,method=key
        for metric in numeric:
            summaries.append({'start':start,'selection':selection,'seed_index':seed,'method':method,'metric':metric,
                **describe([r.get(metric) for r in group])})
    comparisons=[];seed_count=len(profile['training_seed_sets'])
    def values(start,selection,seed,method,metric):
        group=groups.get((start,None,None,'BASE') if method=='BASE' else (start,selection,seed,method),[])
        # Failed cases stay in rows/ledger, but cannot be silently selected into inference.
        return {r['id']:r.get(metric) for r in group if r['completed_job']}
    for start in ('tikhonov','tv'):
        for selection in plan.SELECTIONS:
            for left,right in analysis['comparisons']:
                for metric in analysis['paired_metrics']:
                    lefts=[values(start,selection,s,left,metric) for s in range(seed_count)]
                    rights=[values(start,selection,s,right,metric) for s in range(seed_count)]
                    for seed in [*range(seed_count),'mean_over_seeds']:
                        if isinstance(seed,int):a,b=lefts[seed],rights[seed]
                        else:
                            a={};b={}
                            for name in ids:
                                av=[v.get(name) for v in lefts];bv=[v.get(name) for v in rights]
                                if all(v is not None and np.isfinite(v) for v in av+bv):
                                    # Mean paired differences before object bootstrap, algebraically equivalent to paired means.
                                    a[name]=float(np.mean(np.asarray(av)-np.asarray(bv)));b[name]=0.
                        comparisons.append({'start':start,'selection':selection,'seed_index':seed,'left':left,'right':right,'metric':metric,
                            **paired_effect(a,b,ids,seed=analysis['bootstrap']['seed'],resamples=analysis['bootstrap']['resamples'],higher_is_better=metric in ('ssim','psnr'))})
            # Descriptive seed mean per object, never concatenate seeds as observations.
            for method in plan.METHODS:
                for metric in numeric:
                    maps=[values(start,selection,s,method,metric) for s in range(seed_count)]
                    vals=[float(np.mean([m[i] for m in maps])) if all(i in m and m[i] is not None and np.isfinite(m[i]) for m in maps) else None for i in ids]
                    summaries.append({'start':start,'selection':selection,'seed_index':'mean_over_seeds','method':method,'metric':metric,**describe(vals)})
    # All methods against BASE and NSN (descriptive error/complement changes included).
    descriptive=[]
    for start in ('tikhonov','tv'):
        for selection in plan.SELECTIONS:
            for seed in range(seed_count):
                for method in plan.METHODS:
                    for reference in ('BASE','NSN'):
                        if reference==method:continue
                        for metric in ('mse','ssim','psnr','relative_l2','error_null_mse','error_range_mse','error_cross_term'):
                            a=values(start,selection,seed,method,metric);b=values(start,selection,seed,reference,metric)
                            valid=[i for i in ids if a.get(i) is not None and b.get(i) is not None]
                            differences=[a[i]-b[i] for i in valid]
                            descriptive.append({'start':start,'selection':selection,'seed_index':seed,'method':method,'reference':reference,'metric':metric,
                                'missing_ids':sorted(set(ids)-set(valid)),'worsened_fraction':float(np.mean([d<0 if metric in ('ssim','psnr') else d>0 for d in differences])) if differences else None,**describe(differences)})
    budget=[]
    for key,group in groups.items():
        start,selection,seed,method=key
        budget.append({'start':start,'selection':selection,'seed_index':seed,'method':method,'count':len(group),
            'role':'reference budget only' if method=='RES' else 'leakage-corrected constraint',
            'raw_beta_exceedances':sum(r.get('signed_beta_excess') is not None and r['signed_beta_excess']>0 for r in group),
            'raw_bound_exceedances':sum(r.get('signed_bound_excess') is not None and r['signed_bound_excess']>0 for r in group),
            'tolerated_bound_violations':sum(r.get('constraint_violation') is True for r in group),
            'beta_exceedances_after_tolerance':sum(r.get('beta_exceeded_with_tolerance') is True for r in group),
            'nsn_leakage_identity_failures':sum(r.get('nsn_leakage_identity_passed') is False for r in group),
            'reference_delta_exceedances':sum(r['reference_budget_excess']>0 for r in group),
            'clipping_fraction':float(np.mean([r['clipping_active'] for r in group if r['clipping_active'] is not None])) if any(r['clipping_active'] is not None for r in group) else None,
            'max_beta_excess':max((r['signed_beta_excess'] for r in group if r['signed_beta_excess'] is not None),default=None),
            'max_bound_excess':max((r['signed_bound_excess'] for r in group if r['signed_bound_excess'] is not None),default=None),
            'max_decomposition_residual':max((abs(r['error_decomposition_residual']) for r in group if 'error_decomposition_residual' in r),default=None)})
    output=root/'reports'/(phase+'_'+utc_now().replace(':','').replace('.','_'));output.mkdir(parents=True,exist_ok=False)
    for name,data in [('individual_values',rows),('descriptive',summaries),('paired_effects',comparisons),('changes_against_BASE_NSN',descriptive),('budget',budget),('case_ledger',ledger),('attempt_ledger',attempts)]:
        write_json(output/(name+'.json'),data);table(output/(name+'.csv'),data)
    plots=[]
    for start in ('tikhonov','tv'):
        for selection in plan.SELECTIONS:
            prefix=start+'_'+selection
            effects=[r for r in comparisons if r['start']==start and r['selection']==selection and r['seed_index']=='mean_over_seeds' and r['metric']=='mse']
            series=[(r['left']+' - '+r['right'],[(i,r['summary']['mean'])],[r['bca']['interval']]) for i,r in enumerate(effects) if 'mean' in r['summary']]
            name=prefix+'_paired_quality.svg';chart(output/name,'Paired MSE effect: 95% BCa, mean over seeds',series,'comparison','MSE difference');plots.append(name)
            series=[];decomposition=[];data_series=[]
            for method in ('BASE',*plan.METHODS):
                group=[r for r in rows if r['start_kind']==start and r['method']==method and (r['selection']==selection or method=='BASE')]
                series.append((method,[(r['data_change_over_noise'],r['mse']) for r in group if r['data_change_over_noise'] is not None]))
                decomposition.append((method,[(i,float(np.mean([r[k] for r in group]))) for i,k in enumerate(('error_null_mse','error_range_mse','error_cross_term'))] if group else []))
                data_series.append((method,[(i,float(np.mean([r[k] for r in group]))) for i,k in enumerate(('start_residual','data_residual','delta_noise','delta_effective','model_error'))] if group else []))
            for suffix,title,s,x,y in [('quality_data','Quality versus data change',series,'||A(xhat-z)|| / delta','MSE'),('error_decomposition','Numerical error decomposition',decomposition,'0: E_N, 1: E_R, 2: cross term','pixel MSE contribution'),('data_diagnostics','Data-space diagnostics',data_series,'0: r0, 1: r, 2: noise, 3: effective, 4: discretization','weighted data norm')]:
                name=prefix+'_'+suffix+'.svg';chart(output/name,title,s,x,y);plots.append(name)
            series=[]
            for method in ('NSN','DP','DP_TWO'):
                entries=[r for r in budget if r['start']==start and r['selection']==selection and r['method']==method]
                series.append((method,[(i,max(r[k] for r in entries if r[k] is not None)) for i,k in enumerate(('max_beta_excess','max_bound_excess'))] if entries else []))
            name=prefix+'_budget.svg';chart(output/name,'Maximum signed budget excess (raw, before tolerance)',series,'0: versus beta, 1: versus leakage-corrected bound','weighted data norm');plots.append(name)
    curves=[];resources=[]
    for job in read(root/'validation_jobs.json'):
        if job['kind']!='train':continue
        events=[]
        for run,limit in training_segments(root,job):
            marker=run/'completion.json'
            starts={}
            for row in jsonlines(run/'evaluations.jsonl'):
                event=row['evaluation_id']
                if 'summary' not in row:starts[event]=row;continue
                merged={**starts.get(event,{}),**row}
                if merged.get('weight_reference') is None:continue
                if limit and (merged['step']>limit[0] or (merged['step']==limit[0] and limit[1]=='validate')):continue
                events.append(merged)
            resources.append({'job':job['id'],'run':str(run),'completion':file_reference(marker),
                'summary':file_reference(run/'summary.json') if (run/'summary.json').exists() else None,
                'environment':file_reference(run/'environment.json'),
                'step_log':file_reference(run/'training_steps.jsonl') if (run/'training_steps.jsonl').exists() else None})
        curves.append({'job':job['id'],'events':events})
    write_json(output/'learning_curves.json',curves);write_json(output/'resource_logs.json',resources)
    for start in ('tikhonov','tv'):
        series=[]
        for row in curves:
            if '_'+start+'_' not in row['job']:continue
            points=[]
            for e in row['events']:
                summary=e.get('summary',{})
                mse=summary.get('mse',{});value=mse.get('mean') if isinstance(mse,dict) else None
                if value is not None:points.append((e.get('step',0),value))
            series.append((row['job'],sorted(set(points))))
        name=start+'_learning.svg';chart(output/name,'Validation learning curves',series,'optimizer step','MSE');plots.append(name)
        loss_series=[]
        for job in read(root/'validation_jobs.json'):
            if job['kind']!='train' or job['start']!=start:continue
            points={}
            for run,limit in training_segments(root,job):
                for row in jsonlines(run/'training_steps.jsonl'):
                    if limit and row['step']>limit[0]:continue
                    points[row['step']]=row['loss']
            loss_series.append((job['id'],sorted(points.items())))
        name=start+'_training_loss.svg';chart(output/name,'Training loss; resumed steps retain their archive lineage',loss_series,'optimizer step','batch MSE');plots.append(name)
    panels=make_panels(output,rows,read(root/'protocol_design.json')['example_ids'][phase],analysis)
    expected_evals=sum(j['kind']=='evaluate' for j in jobs)
    complete=not issues and len(rows)==expected_evals*len(ids)
    report={'schema':1,'kind':profile['kind'],'phase':phase,'complete':complete,'issues':issues,
        'expected_jobs':len(jobs),'expected_evaluation_groups':expected_evals,'expected_rows':expected_evals*len(ids),
        'observed_rows':len(rows),'unique_objects':len(ids),'training_seeds':profile['training_seed_sets'],
        'definition':file_reference(root/'definition.json'),'inputs':file_reference(root/'inputs.json'),
        'implementation':implementation(),'analysis_seed':analysis['bootstrap']['seed'],'bootstrap_resamples':analysis['bootstrap']['resamples'],
        'input_references':refs,'gpu_count':len([d for d in devices if d.startswith('GPU-')]),'devices':sorted(devices),
        'resume_exercised':len(resumed)==len(profile['training_seed_sets'])*8,'resumed_jobs':sorted(resumed),
        'base_counting':'One BASE result per start/object; shared references, never extra independent observations',
        'interpretation':analysis['interpretation'],'all_attempts_retained':True,'scientific_main_run':profile['kind']=='main'}
    write_json(output/'report.json',report)
    contents=['<!doctype html><meta charset="utf-8"><title>DPNN study report</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>DPNN '+html.escape(profile['kind']+' / '+phase)+'</h1>',
        '<p>Complete: '+str(complete)+'. Rows: '+str(len(rows))+' / '+str(report['expected_rows'])+'.</p>',
        '<p>MSE is primary; best_mse and best_ssim remain separate. PSNR is calculated per object with range 1 before averaging. No metric clipping or normalization. Smaller residuals do not establish image quality. BCa intervals describe object uncertainty conditional on these models; missing pairs and constant effects are explicitly identified. Seeds are not extra images. Numerical projector components are not continuous missing-wedge components.</p>',
        '<p>Image display uses common [0,1] limits; stored arrays and all metrics are unclipped. Solver durations are in source solver records; cache-load duration is not solver time. Parameter counts, forward/diagnostics times and memory/environment logs are linked below.</p>',
        '<p>Intervals are prespecified comparisons without multiplicity adjustment. See <a href="report.json">report.json</a> for the complete provenance and <a href="resource_logs.json">resource_logs.json</a> for timing/memory logs.</p>',
        '<pre>'+html.escape('\n'.join(issues) if issues else 'All expected cases present; no duplicate object rows or failed constraints.')+'</pre>',
        '<h2>Image quality</h2><table border="1" cellpadding="5"><tr><th>Start / selection / seed / method</th><th>Metric</th><th>Mean ± population SD</th><th>Median</th><th>n</th></tr>']
    for row in summaries:
        if row['metric'] not in ('mse','ssim','psnr','relative_l2') or 'mean' not in row:continue
        label=' / '.join(str(row[k]) for k in ('start','selection','seed_index','method'))
        contents.append(f'<tr><td>{html.escape(label)}</td><td>{row["metric"]}</td><td>{row["mean"]:.6g} ± {row["population_std"]:.4g}</td><td>{row["median"]:.6g}</td><td>{row["count"]}</td></tr>')
    contents.append('</table>')
    for path in sorted(output.glob('*.csv')):contents.append(f'<p><a href="{path.name}">{path.name}</a></p>')
    for name in plots:contents.append(f'<p><a href="{name}"><img src="{name}"></a></p>')
    for item in panels:contents.append('<p>'+html.escape(item['label'])+f'<br><img src="{item["file"]}"></p>')
    (output/'report.html').write_text('\n'.join(contents))
    write_json(output/'artifact_index.json',{'files':[file_reference(p) for p in sorted(output.iterdir()) if p.is_file()]})
    write_json(root/(phase+'_report.json'),file_reference(output/'report.json'))
    return output/'report.json'


def make_panels(output,rows,fixed,analysis):
    panels=[]
    for start in ('tikhonov','tv'):
        for selection in plan.SELECTIONS:
            seeds=sorted({r['seed_index'] for r in rows if r['seed_index'] is not None})
            for seed in seeds:
                selected=[r for r in rows if r['start_kind']==start and (r['method']=='BASE' or (r['selection']==selection and r['seed_index']==seed))]
                lookup={(r['id'],r['method']):r for r in selected}
                difficult=set()
                for method in plan.METHODS:
                    difficult.update(r['id'] for r in sorted((r for r in selected if r['method']==method),key=lambda r:r['mse'],reverse=True)[:analysis['examples']['difficult_count']])
                for name in [*fixed,*sorted(difficult-set(fixed))]:
                    available=[lookup.get((name,m)) for m in ('BASE',*plan.METHODS)]
                    if not all(available):continue
                    images=[]
                    for r in available:
                        with np.load(verify_reference(r['prediction']),allow_pickle=False) as data:
                            if not images:images.append(('target',data['target'].squeeze().copy()))
                            images.append((r['method'],data['prediction'].squeeze().copy()))
                    size=images[0][1].shape[0];canvas=Image.new('RGB',(len(images)*size,size+28),'white');draw=ImageDraw.Draw(canvas)
                    for k,(label,array) in enumerate(images):
                        image=Image.fromarray(np.uint8(np.clip(array,0,1)*255),'L').convert('RGB');canvas.paste(image,(k*size,28));draw.text((k*size+2,4),label,fill='black')
                    filename=f'panel_{len(panels):04d}.png';canvas.save(output/filename)
                    label=f'{start}, {selection}, seed {seed}, {name}; '+('predeclared example' if name in fixed else 'post-hoc difficult case: largest method MSE')
                    panels.append({'file':filename,'label':label,'id':name,'display_limits':[0,1],'metric_clipping':False})
    write_json(output/'image_panels.json',panels);return panels
