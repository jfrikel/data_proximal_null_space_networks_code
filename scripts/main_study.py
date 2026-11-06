#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Explicit preparation, validation, freeze, test, report; no phase autorun."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dpnn_experiments import study_plan as plan
from dpnn_experiments import study_queue as queue
from dpnn_experiments.main_integrity import read
from dpnn_experiments.provenance import write_json
from dpnn_experiments.training_archive import file_reference, verify_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','bind-inputs','prepare-inputs','start','resume','status','stop','freeze-tests','report','review-rehearsal','_run','_attempt'])
    p.add_argument('root',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--phase',choices=['validation','test'],default='validation');p.add_argument('--cpu-fixture',action='store_true')
    p.add_argument('--foreground',action='store_true');p.add_argument('--attempt',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args();root=a.root.resolve();command=a.command
    if command=='prepare':result=str(plan.prepare(read(a.profile),root))
    elif command=='bind-inputs':
        with queue.locked(root/'inputs.lock'):result=plan.bind_inputs(root,read(a.inputs))
    elif command=='prepare-inputs':result=queue.prepare_inputs(root)
    elif command in ('start','resume'):result=queue.start(root,a.phase,a.cpu_fixture,a.foreground)
    elif command=='_run':result=queue.run(root,a.phase,a.cpu_fixture)
    elif command=='_attempt':
        spec=read(a.attempt/'dispatch.json');queue.execute_attempt(root,spec['job'],a.attempt);return
    elif command=='status':
        result={'readiness':read(root/'readiness.json'),'status':read(root/'status.json') if (root/'status.json').exists() else 'not_started',
            'controller_alive':queue.controller_alive(root),
            'inputs_bound':(root/'inputs.json').exists(),'tests_frozen':(root/'test_protocol_index.json').exists()}
    elif command=='stop':result=queue.stop(root)
    elif command=='freeze-tests':
        with queue.locked(root/'controller.lock'):result=str(plan.freeze_tests(root))
    elif command=='report':
        from dpnn_experiments.study_report import build
        with queue.locked(root/'controller.lock'):result=str(build(root,a.phase))
    elif command=='review-rehearsal':
        report=read(a.report)
        if report.get('kind')!='rehearsal' or not report.get('complete') or report.get('phase')!='test':raise ValueError('Complete technical rehearsal report required')
        if report.get('implementation')!=plan.implementation() or report.get('gpu_count')!=4 or not report.get('resume_exercised'):raise ValueError('Current four-GPU/resume qualification required')
        result={'accepted':True,'definition':file_reference(root/'definition.json'),'report':file_reference(a.report),'review':'Explicit review-rehearsal CLI action after inspection'}
        for ref in [report['definition'],report['inputs'],*report['input_references']]:verify_reference(ref)
        rehearsal_definition=read(verify_reference(report['definition']))
        rehearsal_template=read(verify_reference(rehearsal_definition['files']['training_template']))
        main_template=read(root/'training_template.json')
        for key in ('geometry','computation','projection','model','optimizer','scheduler','selection','metrics','logging'):
            if rehearsal_template[key]!=main_template[key]:raise ValueError('Rehearsal/main numerical contract differs: '+key)
        write_json(root/'rehearsal_review.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
