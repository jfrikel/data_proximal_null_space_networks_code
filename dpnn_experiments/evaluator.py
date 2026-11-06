# SPDX-License-Identifier: GPL-3.0-only
"""Standalone frozen validation/test evaluator with certified start-solver stops.

Does not select models or update parameters. Beta overrides are evaluation-only.
Every source dataset, checkpoint and locked protocol is hash-bound in the manifest.
"""
import argparse,copy,csv,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from dpnn_experiments.projections import MatrixOperator
from dpnn_experiments.reconstruction import reconstruct
from dpnn_experiments.projection_acceptance import accept_projection
from dpnn_experiments.training import build_model, build_correction_model, ssim
from dpnn_experiments.projection_diagnostics import setup, array
from dpnn_experiments.training_archive import TrainingArchive, file_reference, verify_reference, canonical_hash
from dpnn_experiments.provenance import write_json, digest

from dpnn_experiments.cached_starts import EvaluationStartCache

SOURCE_ROOT=Path(__file__).resolve().parents[1]
METRICS={'data_range':1.,'ssim_window':11,'ssim_sigma':1.5,'ssim_k1':.01,'ssim_k2':.03,'ssim_covariance':'population','ssim_padding':'valid'}


def ratio(a,b):return float(a/b) if b>0 else None


def matrix_operator(p):
    a=p.matrix
    if a.is_sparse:
        a=a.coalesce();idx=a.indices();vals=a.values().abs()
        rows=vals.new_zeros(a.shape[0]).scatter_add_(0,idx[0],vals)
        cols=vals.new_zeros(a.shape[1]).scatter_add_(0,idx[1],vals)
        bound=float(rows.max()*cols.max())
    else:bound=float(a.abs().sum(0).max()*a.abs().sum(1).max())
    return MatrixOperator(a,bound)


def protocol(c):
    keys=('dataset','geometry','model_config','models','starts','beta_multipliers','split','ids','metrics','tolerances','diagnostic_ids','computation','allow_start_transfer')
    result={k:copy.deepcopy(c[k]) for k in keys}
    for key in ('analysis_policy','include_baseline','baseline_reference'):
        if key in c:result[key]=copy.deepcopy(c[key])
    return result


def validate(c):
    if c['split'] not in ('validation','test'):raise ValueError('Evaluation split must be validation or test')
    if not c['ids'] or len(c['ids'])!=len(set(c['ids'])):raise ValueError('Unique nonempty evaluation IDs required')
    if not set(c['diagnostic_ids'])<=set(c['ids']):raise ValueError('Diagnostic IDs outside evaluation')
    if c['computation']['dtype']!='float64' or c['computation']['device'] not in ('cpu','cuda:0'):raise ValueError('Only qualified Float64 devices')
    if type(c['allow_start_transfer']) is not bool:raise ValueError('Explicit start-transfer policy required')
    if not c['starts'] or len({x['kind'] for x in c['starts']})!=len(c['starts']):raise ValueError('Unique starts required')
    if any(x['kind'] not in ('tikhonov','tv') for x in c['starts']):raise ValueError('Only Tikhonov and TV starts')
    if not c['beta_multipliers'] or any(not math.isfinite(x) or x<0 for x in c['beta_multipliers']):raise ValueError('Invalid beta sweep')
    if len(set(c['beta_multipliers']))!=len(c['beta_multipliers']):raise ValueError('Duplicate beta values')
    if any(not math.isfinite(v) or v<=0 for v in c['tolerances'].values()):raise ValueError('Positive tolerances required')
    if set(c['tolerances'])!={'atol','rtol'}:raise ValueError('Explicit absolute and relative tolerances required')
    pairs=[(x['method'],x['trained_start']) for x in c['models']]
    if len(pairs)!=len(set(pairs)):raise ValueError('Duplicate method/training-start combination')
    for start in c['starts']:
        for method in {x['method'] for x in c['models']}:
            candidates=[x for x in c['models'] if x['method']==method]
            matches=[x for x in candidates if x['trained_start']==start['kind']]
            if not matches and (not c['allow_start_transfer'] or len(candidates)!=1):raise ValueError('Start transfer not authorized or ambiguous in protocol')
    if c['split']=='test':
        if c['frozen_protocol'] is None:raise ValueError('Test requires pre-existing frozen protocol')
        frozen=json.loads(verify_reference(c['frozen_protocol']).read_text())
        if frozen!={'protocol':protocol(c),'selection_closed':True}:raise ValueError('Frozen test protocol mismatch')


def summarize(rows):
    groups={}
    for row in rows:groups.setdefault(row['condition'],[]).append(row)
    result={}
    for key,rs in groups.items():
        fields={}
        for k in rs[0]:
            vs=[r[k] for r in rs if isinstance(r[k],(int,float)) and not isinstance(r[k],bool)]
            if not vs:continue
            x=np.asarray(vs);fields[k]={'count':len(vs),'mean':float(x.mean()),'median':float(np.median(x)),
                'population_std':float(x.std()),'q05':float(np.quantile(x,.05)),'q95':float(np.quantile(x,.95)),
                'min':float(x.min()),'max':float(x.max())}
        result[key]={'objects':len(rs),'metrics':fields,'constraint_violations':sum(r['constraint_violation'] is True for r in rs),
                     'clipping_count':sum(r['clipping_active'] is True for r in rs)}
    return result


def measurement_metrics(proj,out,start,target,y,clean,raster,delta,beta,details,settings,tolerances,*,separate=False):
    residual=proj.data(out)-y[None];initial=proj.data(start)-y[None]
    change=proj.data(out-start);r=float(residual.norm());r0=float(initial.norm());d=float(change.norm())
    diff=out-target;mse=float(diff.square().mean());eff=float((y-raster).norm())
    row={'mse':mse,'psnr':10*math.log10(settings['data_range']**2/mse) if mse>0 else None,
      'psnr_zero_mse':mse==0,'ssim':float(ssim(out,target,settings)[0]),'relative_l2':ratio(float(diff.norm()),float(target.norm())),
      'data_residual':r,'start_residual':r0,'residual_reduction':r0-r,'relative_residual_reduction':ratio(r0-r,r0),
      'data_change':d,'data_change_over_noise':ratio(d,delta),
      'data_residual_over_measurement':ratio(r,float(y.norm())),
      'delta_noise':delta,'delta_effective':eff,'model_error':float((raster-clean).norm()),
      'residual_over_noise':ratio(r,delta),'residual_over_effective':ratio(r,eff),
      'beta':beta,'change_over_beta':ratio(d,beta) if beta is not None else None,
      'correction_norm':float((out-start).norm())*proj.data_scale,'correction_data_norm':d,
      'constraint_violation':None,'clipping_active':None,'control':None,'null_leak':None,'leakage_bound':None,
      'signed_beta_excess':None,'signed_bound_excess':None,'constraint_tolerance':None}
    if details is not None:
        value,null,ran,t=details;ell=float(proj.data(null).norm())
        row.update(raw_correction_norm=float(value.norm())*proj.data_scale,
            raw_correction_data_norm=float(proj.data(value).norm()),
            null_norm=float(null.norm())*proj.data_scale,range_norm=float(ran.norm())*proj.data_scale,
            applied_range_norm=abs(t)*float(ran.norm())*proj.data_scale,null_leak=ell,control=t)
        if beta is not None:
            bound=beta+(1 if separate else 1-t)*ell
            # Data-scale-aware rounding allowance; no unitless floor at one.
            tol=tolerances['atol']+tolerances['rtol']*max(float(proj.data(out).norm()),float(proj.data(start).norm()),bound)
            row.update(leakage_bound=bound,signed_beta_excess=d-beta,signed_bound_excess=d-bound,
                       constraint_tolerance=tol,constraint_violation=d>bound+tol,clipping_active=t<1 if beta>0 else False)
    return row,{'residual':array(residual),'start_residual':array(initial),'data_change':array(change)}


def error_decomposition(projection, prediction, target, tolerances):
    """Pixel-MSE decomposition with exactly the bound numerical P, no h scaling."""
    error=prediction-target
    null,ran=projection.components(error)
    en=float(null.square().mean());er=float(ran.square().mean())
    cross=float(2*(null*ran).mean());mse=float(error.square().mean())
    discrepancy=mse-en-er-cross
    tolerance=tolerances['atol']+tolerances['rtol']*max(mse,en+er+abs(cross))
    return {'error_null_mse':en,'error_range_mse':er,'error_cross_term':cross,
        'error_decomposition_residual':discrepancy,'error_decomposition_tolerance':tolerance,
        'error_decomposition_passed':abs(discrepancy)<=tolerance}, {'error_null':array(null),'error_range':array(ran)}


def execute(c,root):
    c=copy.deepcopy(c);root=Path(root);c['training']={'applicable':False,'reason':'Frozen evaluation only; no optimizer or model selection'}
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with TrainingArchive(c,root/'runs',SOURCE_ROOT,root/'metadata') as a:
        print('Evaluation archive:',a.path,flush=True);validate(c)
        binding=json.loads(verify_reference(c['dataset']['reference']).read_text())
        splits=binding['split_ids'];flat=[x for v in splits.values() for x in v]
        if len(flat)!=len(set(flat)) or not set(c['ids'])<=set(splits[c['split']]):raise ValueError('Split violation')
        records={r['id']:r for r in binding['records']}
        if len(records)!=len(flat) or set(records)!=set(flat):raise ValueError('Dataset ID mismatch')
        if binding['geometry']!=c['geometry'] or c['model_config']['geometry']!=c['geometry']:raise ValueError('Geometry mismatch')
        pc=c['model_config']['projection']
        if pc['kind']=='block' and binding['operator_sha256']!=pc['operator_sha256']:raise ValueError('Operator mismatch')
        write_json(a.path/'dataset_binding.json',binding);write_json(a.path/'effective_protocol.json',protocol(c))
        caches={s['kind']:EvaluationStartCache(s,binding,c['geometry'],c['ids'],c['split']) for s in c['starts'] if 'cache' in s}
        for kind,cache in caches.items():write_json(a.path/f'start_cache_{kind}_binding.json',cache.binding)
        write_json(a.path/'metric_definitions.json',{'image_norm':'h times Euclidean norm of physical pixels; MSE is unweighted pixel mean','data_norm':'Euclidean norm of sqrt(wY) weighted measurements','TV':'isotropic Neumann forward differences, h sum |Dx|; unconstrained','null_ratios':'null means zero denominator; PSNR null for exact zero MSE','budget':'additional data change, not total residual','tolerance':'atol+rtol*max(norm(A output),norm(A start),budget_with_leak)','test_policy':'no optimizer, no checkpoint selection; locked exact protocol required','training_beta':'recorded in original training manifest and its hash-bound dataset; evaluation override never modifies it'})
        write_json(a.path/'correction_and_timing_definitions.json',{
            'correction_norm':'h*norm(output-start), including both branches for DP_TWO',
            'correction_data_norm':'norm(A(output-start)); equals data_change',
            'raw_correction_norm':'h*norm(U) for RES/NSN/DP; h*norm(V) for DP_TWO',
            'null_branch_correction_norm':'h*norm(U) for DP_TWO',
            'null_norm':'h*norm(PU) before addition to the start',
            'range_norm':'h*norm((I-P)U) or h*norm((I-P)V), before clipping',
            'applied_range_norm':'abs(control)*range_norm; zero for NSN',
            'inference_seconds':'Synchronized model forward only; weights preloaded; no extra RES projection',
            'diagnostics_seconds':'Extra decompositions, image metrics and constraint diagnostics after forward'})
        torch.set_num_threads(c['computation']['cpu_threads']);device=setup(a,c)
        model=build_model(c['model_config'],device,a);model.eval();p=model.projection;op=matrix_operator(p)
        accept_projection(p,a,c['randomness']['seed'])
        # Preload and verify provenance before reading any evaluation samples.
        networks=[]
        for entry in c['models']:
            if entry['method'] not in ('RES','NSN','DP','DP_TWO'):raise ValueError('Unknown model method')
            sourcepath=verify_reference(entry['training_manifest']);source=json.loads(sourcepath.read_text());sc=source['configuration']
            completion=json.loads((sourcepath.parent/'completion.json').read_text())
            if completion['status']!='completed' or completion['manifest_sha256']!=digest(sourcepath) or completion['artifacts_sha256']!=digest(sourcepath.parent/'artifacts.json'):raise ValueError('Invalid training archive')
            index=json.loads((sourcepath.parent/'artifacts.json').read_text())['files']
            if not any(x['sha256']==entry['weights']['sha256'] and x['bytes']==entry['weights']['bytes'] and x['path'].startswith('weights/') for x in index):raise ValueError('Weights not in source archive')
            if entry['trained_start'] not in ('tikhonov','tv'):raise ValueError('Unknown training start')
            expected=copy.deepcopy(c['model_config']['model']);expected['method']=entry['method']
            if sc['model']!=expected:raise ValueError('Architecture mismatch')
            if sc['model']['method']!=entry['method'] or sc['model']['architecture']!=c['model_config']['model']['architecture']:raise ValueError('Model provenance mismatch')
            if sc['geometry']!=c['geometry'] or sc['projection']!=pc:raise ValueError('Model operator mismatch')
            if c['split']=='test' and set(c['ids'])&set(sc['dataset']['used_ids']['train']+sc['dataset']['used_ids']['validation']):raise ValueError('Test IDs used for training/selection')
            if entry['trained_start'] in caches and sc['dataset']['reference']!=caches[entry['trained_start']].reference:
                raise ValueError('Training/evaluation start cache mismatch')
            weightpath=verify_reference(entry['weights']);state=torch.load(weightpath,map_location=device,weights_only=True)
            cfg=copy.deepcopy(c['model_config']);cfg['model']['method']=entry['method']
            net=model if not networks and entry['method']==c['model_config']['model']['method'] else build_correction_model(cfg,p,device).eval()
            net.load_state_dict(state,strict=True)
            networks.append((entry,net))
            del state
        rows=[];size=c['geometry']['size']
        parameter_counts={e['method']:sum(x.numel() for x in net.correction.parameters()) for e,net in networks}
        with torch.no_grad():
            for index,name in enumerate(c['ids']):
                record=records[name]
                if record['split']!=c['split']:raise ValueError('Record split mismatch')
                with np.load(verify_reference(record['file']),allow_pickle=False) as z:
                    data={k:torch.tensor(z[k],device=device,dtype=torch.float64) for k in ('target','measurement','analytic_clean','raster_clean','delta')}
                target=data['target'][None];y=data['measurement'];delta=float(data['delta'])
                if delta<0:raise ValueError('Negative noise norm')
                expected_raster=p.data(target)[0]
                if not torch.allclose(expected_raster,data['raster_clean'],atol=1e-10,rtol=1e-10):raise ValueError('Raster/operator data mismatch')
                if not all(bool(torch.isfinite(v).all()) for v in data.values()):raise ValueError('Nonfinite sample')
                if not math.isclose(float((y-data['analytic_clean']).norm()),delta,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Noise metadata mismatch')
                for start_settings in c['starts']:
                    kind=start_settings['kind'];started=time.perf_counter()
                    if kind in caches:
                        start,solver=caches[kind].load(name,record,device)
                    else:
                        start,solver=reconstruct(op,y,size,start_settings)
                    if device.type=='cuda':torch.cuda.synchronize()
                    solver['wall_seconds']=time.perf_counter()-started
                    write_json(a.path/f'solvers/{index:05d}_{kind}.json',solver)
                    if not solver['passed']:raise ArithmeticError('Start solver did not meet recorded stopping rule')
                    start=start[None,None];conditions=[('BASE',None,None,None)] if c.get('include_baseline',True) else []
                    selected=[]
                    for method in sorted({e['method'] for e,state in networks}):
                        candidates=[(e,state) for e,state in networks if e['method']==method]
                        matched=[(e,state) for e,state in candidates if e['trained_start']==kind]
                        selected.append((matched or candidates)[0])
                    conditions.extend((e['method'],e,state,mult) for e,state in selected for mult in (c['beta_multipliers'] if e['method'] in ('DP','DP_TWO') else [None]))
                    for method,entry,state,mult in conditions:
                        if device.type=='cuda':torch.cuda.synchronize(device)
                        inference_started=time.perf_counter()
                        value=null=ran=torch.zeros_like(start);t=0.;beta=0. if method in ('BASE','NSN') else None
                        if state is None:out=start
                        elif method in ('DP','DP_TWO'):
                            beta=mult*delta;out,dd=state(start,beta,return_details=True)
                        else:
                            out,dd=state(start,return_details=True)
                        if device.type=='cuda':torch.cuda.synchronize(device)
                        inference_seconds=time.perf_counter()-inference_started
                        diagnostics_started=time.perf_counter()
                        if state is not None:
                            value=dd['value']
                            if method=='RES':null,ran=p.components(value)
                            else:null,ran=dd['null'],dd['range']
                            t=float(dd['control'][0]) if method in ('DP','DP_TWO') else dd['control']
                        detail=(value,null,ran,t)
                        row,arrays=measurement_metrics(p,out,start,target,y,data['analytic_clean'],data['raster_clean'],delta,beta,detail,c['metrics'],c['tolerances'],separate=method=='DP_TWO')
                        if c.get('analysis_policy',{}).get('error_decomposition'):
                            decomposition,error_arrays=error_decomposition(p,out,target,c['tolerances'])
                            row.update(decomposition);arrays.update(error_arrays)
                        row['bound_formula']='beta+norm(APU)' if method=='DP_TWO' else 'beta+(1-t)*norm(APU)'
                        row['budget_role']='reference_only' if method=='RES' else ('unchanged_start' if method=='BASE' else 'certified_with_leakage')
                        row['reference_budget_excess']=row['data_change']-delta
                        row['nsn_change_minus_leak']=row['data_change']-row['null_leak'] if method=='NSN' else None
                        row['nsn_leakage_identity_passed']=abs(row['nsn_change_minus_leak'])<=row['constraint_tolerance'] if method=='NSN' else None
                        row['beta_exceeded_with_tolerance']=row['signed_beta_excess']>row['constraint_tolerance'] if row['signed_beta_excess'] is not None else None
                        if method=='DP_TWO':row['null_branch_correction_norm']=float(dd['null_value'].norm())*p.data_scale
                        row['raw_correction_branch']='V' if method=='DP_TWO' else ('none' if method=='BASE' else 'U')
                        if device.type=='cuda':torch.cuda.synchronize(device)
                        row['diagnostics_seconds']=time.perf_counter()-diagnostics_started
                        label=f'{kind}_{method}'+(f'_beta{mult:g}' if mult is not None else '')
                        row.update(inference_seconds=inference_seconds,
                            start_solver_seconds=None if kind in caches else solver['wall_seconds'],
                            start_load_seconds=solver['wall_seconds'] if kind in caches else None,
                            original_cached_solver_seconds=solver.get('original_solver',{}).get('wall_seconds') if kind in caches else None,
                            parameter_count=parameter_counts.get(method,0),
                            id=name,split=c['split'],noise_realization_id=record['noise_realization_id'],condition=label,method=method,start_kind=kind,beta_multiplier=mult,
                            weights_sha256=entry['weights']['sha256'] if entry else None,
                            trained_start=entry['trained_start'] if entry else None,
                            start_transfer=entry['trained_start']!=kind if entry else False,
                            training_manifest_sha256=entry['training_manifest']['sha256'] if entry else None)
                        path=a.path/f'predictions/{index:05d}_{label}.npz';path.parent.mkdir(exist_ok=True)
                        payload={'prediction':array(out),'start':array(start),'target':array(target)}
                        if name in c['diagnostic_ids']:payload.update(arrays,value=array(value),null=array(null),range=array(ran))
                        if method=='DP_TWO' and name in c['diagnostic_ids']:payload['null_value']=array(dd['null_value'])
                        np.savez_compressed(path,**payload);row['prediction']=file_reference(path)
                        rows.append(row);a.rows('metrics',[row])
                        if row['constraint_violation'] and not c.get('analysis_policy',{}).get('retain_violations'):
                            raise ArithmeticError('Data proximity violated')
                a.flush_metadata();print('Evaluated',name,flush=True)
        with (a.path/'metrics.csv').open('x',newline='') as f:
            fields=sorted(set().union(*(r.keys() for r in rows)));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
        violations=sum(r['constraint_violation'] is True or r.get('error_decomposition_passed') is False or r.get('nsn_leakage_identity_passed') is False for r in rows)
        if violations:a.outcome='failed';a.reason='Retained all evaluated objects; budget/decomposition violations'
        write_json(a.path/'summary.json',{'passed':violations==0,'violations':violations,'parameter_counts':parameter_counts,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,'groups':summarize(rows),'evaluated_ids':c['ids'],'split':c['split'],
            'selection_performed':False,'test_objects_read':len(c['ids']) if c['split']=='test' else 0})
    return a.path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True,type=Path);ap.add_argument('--root',required=True,type=Path)
    x=ap.parse_args();execute(json.loads(x.config.read_text()),x.root)
if __name__=='__main__':main()
