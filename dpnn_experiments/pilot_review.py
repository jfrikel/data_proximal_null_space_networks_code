# SPDX-License-Identifier: GPL-3.0-only
"""Review existing pilot archives without inference, selection, or test-image access."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np

from dpnn_experiments.provenance import RunArchive, digest, write_json
from dpnn_experiments.training_archive import file_reference, verify_reference

SOURCE = Path(__file__).resolve().parents[1]
METHODS = ('RES', 'NSN', 'DP', 'DP_TWO')


def read(path):
    return json.loads(Path(path).read_text())


def audit(root, full=False):
    """Check sealed index, all member sizes and optionally every member hash."""
    completion = read(root/'completion.json')
    if completion['status'] != 'completed':
        raise ValueError('Incomplete archive: '+str(root))
    for key, name in [('manifest_sha256', 'manifest.json'), ('artifacts_sha256', 'artifacts.json')]:
        if completion[key] != digest(root/name):
            raise ValueError('Changed archive seal: '+str(root))
    checked = total = 0
    for item in read(root/'artifacts.json')['files']:
        p = (root/item['path']).resolve()
        if root.resolve() not in p.parents or not p.is_file() or p.stat().st_size != item['bytes']:
            raise ValueError('Invalid archive member: '+str(p))
        total += item['bytes']
        if full or p.suffix in ('.json', '.jsonl', '.csv'):
            verify_reference({'path': str(p), 'bytes': item['bytes'], 'sha256': item['sha256']})
            checked += 1
    return {'run': str(root), 'members': len(read(root/'artifacts.json')['files']),
            'hash_checked_members': checked, 'bytes': total, 'full_hash_audit': full}


def number(row, key):
    return float(row[key]) if row.get(key) not in (None, '') else None


def png(array):
    """Scientific grayscale preview, fixed [0,1] scale; underlying arrays unchanged."""
    x = (np.clip(np.squeeze(array), 0, 1)*255).astype('uint8')
    height, width = x.shape
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data)&0xffffffff)
    data = b''.join(b'\0'+row.tobytes() for row in x)
    return base64.b64encode(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B', width,height,8,0,0,0,0))
                            +chunk(b'IDAT', zlib.compress(data))+chunk(b'IEND', b'')).decode()


def curves_svg(curves, start, metric):
    palette = ['#0072b2', '#d55e00', '#009e73', '#aa4499']
    data = [(m, curves[start+'_'+m]) for m in METHODS]
    vals = [r[metric] for _, rows in data for r in rows]
    lo, hi = min(vals), max(vals)
    if hi == lo: hi = lo+1
    def point(r):
        return f"{55+r['epoch']/50*620:.2f},{220-(r[metric]-lo)/(hi-lo)*180:.2f}"
    lines = ''.join(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="'
                    +' '.join(point(r) for r in rows)+'"/>'
                    +f'<text x="{70+i*150}" y="255" fill="{c}">{m}</text>'
                    for i, ((m, rows), c) in enumerate(zip(data, palette)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 730 280">'
            f'<rect width="730" height="280" fill="white"/><text x="55" y="20">{start}: validation {metric}</text>'
            f'<path d="M55 35V220H680" fill="none" stroke="#555"/><text x="0" y="45">{hi:.3g}</text>'
            f'<text x="0" y="220">{lo:.3g}</text><text x="570" y="275">epoch 0–50</text>{lines}</svg>')


def run(original, continuation, root, full=False):
    original, continuation = Path(original).resolve(), Path(continuation).resolve()
    cfg = {'schema_version': 1, 'purpose': 'Paired pilot validation review; archived states only',
           'geometry': read(original/'plan.json')['training_configuration']['geometry']
               if 'training_configuration' in read(original/'plan.json') else {'size':256},
           'dataset': {'original_plan':file_reference(original/'plan.json'),
                       'continuation_plan':file_reference(continuation/'plan.json')},
           'randomness': {'analysis_seed': 20261101}, 'computation': {'device':'cpu','dtype':'float64'},
           'full_hash_audit': full, 'diagnostic_policy':'First four validation IDs, fixed before viewing'}
    records = read(original/'evaluation_index.json')+read(continuation/'evaluation_index.json')
    if len(records) != 16:
        raise ValueError('Expected both starts, four methods and two selections')
    with RunArchive(cfg, Path(root)/'runs', SOURCE, Path(root)/'metadata') as archive:
        audited, results, timings, curves, examples, all_rows = {}, [], [], {}, {}, []
        for record in records:
            runpath = Path(record['evaluation_run'])
            verify_reference(record['summary'])
            for path in (runpath, Path(record['training_run'])):
                if str(path) not in audited:
                    print('Audit:',path,flush=True)
                    audited[str(path)] = audit(path, full)
            summary = read(runpath/'summary.json')
            if summary['split'] != 'validation' or summary['test_objects_read']:
                raise ValueError('Pilot review must only use validation')
            with (runpath/'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row['split'] != 'validation': raise ValueError('Unexpected split')
                all_rows.append({**row,'selection':record['selection']})
            train = Path(record['training_run'])
            key = record['start']+'_'+record['method']
            if key not in curves:
                learning = []
                for p in sorted(train.glob('validation_*_summary.json')):
                    step = int(p.name.split('_')[1])
                    learning.append({'epoch':step/512, 'mse':read(p)['mse']['mean'], 'ssim':read(p)['ssim']['mean']})
                curves[key] = learning
                ts = read(train/'summary.json')
                events = [json.loads(line) for line in (train/'events.jsonl').read_text().splitlines()]
                seconds = next(x['elapsed_seconds'] for x in reversed(events) if x['event']=='finished')
                timings.append({'start':record['start'],'method':record['method'],
                    'epochs':ts['position']['epoch'],'seconds':seconds,'seconds_per_epoch':seconds/ts['position']['epoch'],
                    'bytes':audited[str(train)]['bytes'],
                    'stop':'early_stopping' if ts['position']['bad_epochs']>=10 else 'epoch_limit',
                    'test_read_count':ts['test_read_count']})
                if ts['test_read_count']:raise ValueError('Training read test images')
            base = {r['id']:r for r in rows if r['method']=='BASE'}
            for condition, group in summary['groups'].items():
                selected = [r for r in rows if r['condition']==condition]
                if len(selected)!=128 or len({r['id'] for r in selected})!=128:
                    raise ValueError('Expected 128 matched validation objects')
                paired = [number(r,'mse')-number(base[r['id']],'mse') for r in selected]
                fields = ('mse','ssim','psnr','data_residual','start_residual','data_change',
                          'null_norm','range_norm','null_leak','signed_bound_excess')
                results.append({'start':record['start'],'method':record['method'],
                    'selection':record['selection'],'condition':condition,'objects':128,
                    **{k:group['metrics'].get(k,{}).get('mean') for k in fields},
                    'constraint_violations':group['constraint_violations'],
                    'mse_wins_vs_start':sum(d<0 for d in paired),
                    'ssim_wins_vs_start':sum(number(r,'ssim')>number(base[r['id']],'ssim') for r in selected),
                    'residual_increases':sum(number(r,'data_residual')>number(r,'start_residual')+1e-12
                                            +1e-10*max(number(r,'data_residual'),number(r,'start_residual')) for r in selected),
                    'paired_mse_delta_mean':float(np.mean(paired)),
                    'summary':record['summary']})
            if record['selection']=='best_mse':
                for ident in sorted(base)[:4]:
                    method = record['method']
                    row = next(r for r in rows if r['id']==ident and r['method']==method
                               and (method not in ('DP','DP_TWO') or number(r,'beta_multiplier')==1))
                    path = verify_reference(json.loads(row['prediction']))
                    with np.load(path, allow_pickle=False) as arrays:
                        examples.setdefault((record['start'],ident),{}).update({method:png(arrays['prediction']),
                            'target':png(arrays['target']),'start':png(arrays['start'])})
        write_json(archive.path/'audit.json',list(audited.values()))
        write_json(archive.path/'results.json',results)
        write_json(archive.path/'learning_curves.json',curves)
        write_json(archive.path/'timings.json',timings)
        fields = sorted({key for row in all_rows for key in row})
        with (archive.path/'individual_metrics.csv').open('x') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
        headers = ['Start','Method','Selection','MSE','SSIM','Residual','Data change','Violations']
        table = '<tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'
        for r in results:
            if r['condition'].endswith('beta0') or (r['condition'].endswith('BASE') and r['method']!='RES'):continue
            cells=[r['start'], 'BASE' if r['condition'].endswith('BASE') else r['method'],r['selection'],
                   *[f'{r[k]:.6g}' if r[k] is not None else '—' for k in ('mse','ssim','data_residual','data_change')],
                   r['constraint_violations']]
            table+='<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>'
        plots=''
        for start in ('tikhonov','tv'):
            for metric in ('mse','ssim'):
                svg=curves_svg(curves,start,metric);name=f'{start}_{metric}.svg'
                (archive.path/name).write_text(svg);plots+='<img class="curve" src="'+name+'">'
        images=''
        for (start,ident), imgs in sorted(examples.items()):
            images+=f'<h3>{html.escape(start+" / "+ident)}</h3><div class="images">'
            for name in ('target','start',*METHODS):
                images+=f'<figure><img src="data:image/png;base64,{imgs[name]}"><figcaption>{name}</figcaption></figure>'
            images+='</div>'
        (archive.path/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>DPNN pilot review</title>'
            '<style>body{font:16px system-ui;margin:30px;max-width:1500px}table{border-collapse:collapse}'
            'td,th{padding:8px;border:1px solid #ccc}.curve{width:48%}.images{display:flex}figure{margin:4px}'
            'figure img{width:170px;max-width:14vw}figcaption{text-align:center}</style>'
            '<h1>DPNN pilot review</h1><p>One training seed; 128 development validation objects. '
            'Both checkpoint selections reported separately. No independent test evaluation or final method ranking.</p>'
            '<p>DP / DP_TWO shown at beta multiplier 1; beta 0 and individual rows in results.json / individual_metrics.csv. '
            'DP_TWO has twice the network capacity. Data proximity constrains added data change, not total residual.</p>'
            '<table>'+table+'</table><h2>Learning curves</h2>'+plots+
            '<h2>Fixed first four validation objects</h2><p>Display scale [0,1], clipped only for display; '
            'stored numerical reconstructions unchanged.</p>'+images)
        write_json(archive.path/'summary.json',{'passed':True,'evaluation_jobs':16,'training_jobs':8,
            'test_quality_evaluated':False,'full_hash_audit':full,'audited_archives':len(audited),
            'timings':timings,'report':'report.html','scientific_acceptance':'requires author review',
            'findings':['TV NSN lowers MSE but has worse SSIM than the TV start; Tikhonov NSN slightly improves SSIM.',
                        'DP_TWO trails shared DP at MSE-selected checkpoints for both starts.',
                        'These observations do not establish an implementation defect or a general ranking.']})
    return archive.path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--continuation',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--full-audit',action='store_true')
    a=p.parse_args();print(run(a.original,a.continuation,a.root,a.full_audit))
