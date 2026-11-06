"""Strict consumption of hash-bound start_preparation datasets; no solver fallback."""
import json
import numpy as np
import torch
from dpnn_experiments.training_archive import verify_reference
from dpnn_experiments.provenance import digest


class EvaluationStartCache:
    def __init__(self, settings, dataset, geometry, ids, split):
        self.reference=settings['cache']
        path=verify_reference(self.reference)
        self.binding=json.loads(path.read_text())
        b=self.binding
        expected={k:v for k,v in settings.items() if k!='cache'}
        if b.get('start_reconstruction')!=expected:raise ValueError('Cached start settings mismatch')
        if b['geometry']!=geometry or b.get('operator_sha256')!=dataset.get('operator_sha256'):
            raise ValueError('Cached start geometry/operator mismatch')
        if b['split_ids']!=dataset['split_ids']:raise ValueError('Cached start split mismatch')
        self.records={r['id']:r for r in b['records']}
        flat=[x for names in b['split_ids'].values() for x in names]
        if len(flat)!=len(set(flat)) or len(self.records)!=len(b['records']) or set(flat)!=set(self.records):
            raise ValueError('Cached start IDs invalid')
        completion=json.loads((path.parent/'completion.json').read_text())
        if completion['status']!='completed' or completion['manifest_sha256']!=digest(path.parent/'manifest.json') or completion['artifacts_sha256']!=digest(path.parent/'artifacts.json'):
            raise ValueError('Cached start archive incomplete or altered')
        artifacts=json.loads((path.parent/'artifacts.json').read_text())['files']
        if not any(x['path']==path.name and x['sha256']==digest(path) for x in artifacts):
            raise ValueError('Cache dataset not in completed archive')
        self.ids=set(ids);self.split=split;self.size=geometry['size']
        for name in ids:
            rec=self.records[name]
            if rec['split']!=split or name not in b['split_ids'][split]:raise ValueError('Cached record split mismatch')
            verify_reference(rec['file']);solver=json.loads(verify_reference(rec['start_solver']).read_text())
            if solver.get('passed') is not True:raise ValueError('Cached start solver failed')

    def load(self,name,source_record,device):
        if name not in self.ids:raise ValueError('Cache read outside declared IDs')
        rec=self.records[name]
        with np.load(verify_reference(rec['file']),allow_pickle=False) as cached, np.load(verify_reference(source_record['file']),allow_pickle=False) as source:
            for key in ('target','measurement','analytic_clean','raster_clean','delta','beta'):
                if key not in cached or key not in source or not np.array_equal(cached[key],source[key]):
                    raise ValueError('Cached start sample mismatch: '+key)
            start=cached['start'].copy()
        if start.shape!=(1,self.size,self.size) or start.dtype!=np.float64 or not np.isfinite(start).all():
            raise ValueError('Invalid cached start array')
        solver=json.loads(verify_reference(rec['start_solver']).read_text())
        if solver.get('passed') is not True:raise ValueError('Cached start solver failed')
        info={'passed':True,'source':'immutable_cache','dataset':self.reference,
              'sample':rec['file'],'solver_reference':rec['start_solver'],'original_solver':solver}
        return torch.tensor(start[0],device=device,dtype=torch.float64),info
