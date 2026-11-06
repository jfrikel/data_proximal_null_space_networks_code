"""Periodic storage reduction and off-interval recovery, with complete archives."""
import argparse,copy,json
from pathlib import Path
import torch
from dpnn_experiments.training import execute
from dpnn_experiments.training_archive import TrainingArchive, verify_reference
from tests.training_qualification import fixture, compare_runs, tensor_difference, rows
from dpnn_experiments.provenance import write_json
SOURCE=Path(__file__).resolve().parents[1]


def run(root):
    root=Path(root);c=json.loads((SOURCE/'configs/tests/scalable_training_qualification.json').read_text())
    c['purpose']='Retention scaling, legacy numerical equivalence, off-interval stops and hard-failure replay'
    with TrainingArchive(c,root/'runs',SOURCE,root/'metadata') as a:
        ref,projection,splits,error=fixture(a,c)
        c['dataset']['reference']=ref;c['dataset']['used_ids']={k:splits[k] for k in ('train','validation')};c['projection']=projection
        c['logging']['checkpoint_every_steps']=1000
        full=execute(c,root);legacy=copy.deepcopy(c);legacy['logging']['weights_every_step']=True
        old=execute(legacy,root)
        wf=sorted((full/'weights').glob('*.pt'));wo=sorted((old/'weights').glob('*.pt'))
        assert len(wf)==4 and len(wo)==10
        assert len(rows(full/'training_steps.jsonl'))==len(rows(old/'training_steps.jsonl'))==9
        for x,y in zip(rows(full/'training_steps.jsonl'),rows(old/'training_steps.jsonl')):
            for k in ('loss','ids','gradient_norm_before_clip','learning_rates','beta','delta'):assert x[k]==y[k],k
        for p in wf:
            assert tensor_difference(torch.load(p,weights_only=True),torch.load(old/'weights'/p.name,weights_only=True))==0
        comparisons=[]
        for label in ('off_interval_interrupt','failure_after_unsaved_step'):
            part=copy.deepcopy(c)
            part['execution']['stop_after_step' if label=='off_interval_interrupt' else 'injected_io_failure_step']=1 if label=='off_interval_interrupt' else 2
            prior=set((root/'runs').iterdir())
            try:first=execute(part,root)
            except OSError:
                assert label=='failure_after_unsaved_step';first=next(iter(set((root/'runs').iterdir())-prior))
            resume=copy.deepcopy(c);resume['execution']['resume']=json.loads((first/'latest_checkpoint.json').read_text())
            state=torch.load(verify_reference(resume['execution']['resume']),weights_only=True)
            assert state['progress']['step']==(1 if label=='off_interval_interrupt' else 0)
            last=execute(resume,root);comparisons.append({'case':label,**compare_runs(full,first,last,0.)})
        write_json(a.path/'summary.json',{'passed':True,'periodic_run':full.name,'legacy_run':old.name,
            'optimizer_steps':9,'periodic_weight_files':len(wf),'legacy_weight_files':len(wo),
            'identical_losses_and_retained_tensors':True,'comparisons':comparisons})
    return a.path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);args=p.parse_args();print(run(args.root))
