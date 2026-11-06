"""Reproduce paper tables/figures from the public scientific export (no inference)."""
from pathlib import Path
import csv,gzip,lzma,json,hashlib,sys,collections,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dpnn-paper-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parent.parent
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,default=ROOT/'results')
parser.add_argument('--cost-file',type=Path,default=ROOT/'results/data/cost.json')
parser.add_argument('--output-dir',type=Path,default=ROOT/'results/generated')
parser.add_argument('--tables-only',action='store_true')
args=parser.parse_args()
SRC=args.data_root.resolve(); OUTPUT=args.output_dir.resolve()
DATA=SRC/'data'; TAB=OUTPUT/'tables'; FIG=OUTPUT/'figures'
TAB.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
PREVIEW=OUTPUT/'previews';PREVIEW.mkdir(parents=True,exist_ok=True)
def read(p):
    op=gzip.open if str(p).endswith('.gz') else lzma.open if str(p).endswith('.xz') else open
    with op(p,'rt') as f:return json.load(f)
def rows(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt') as f:return list(csv.DictReader(f))
def value(v):
    if v in ('True','False'):return v=='True'
    try:return json.loads(v)
    except ValueError:return v or None
R=[{k:value(v) for k,v in r.items()} for r in rows(DATA/'test/individual_values.csv.gz')]
G=collections.defaultdict(list)
for r in R:G[r['start_kind'],r['selection'],r['seed_index'],r['method']].append(r)
M=['BASE','RES','NSN','DP','DP_TWO']; LABEL=dict(zip(M,['BASE','RES','NSN','DP1','DP2']))
starts=['tikhonov','tv']; ST={'tikhonov':'Tikhonov','tv':'TV'}
def group(st,m,s=0,sel='best_mse'):return G[st,None,None,m] if m=='BASE' else G[st,sel,s,m]
def vals(st,m,key,s=0,sel='best_mse'):return np.array([r[key] for r in group(st,m,s,sel)],float)
def mean(st,m,key,sel='best_mse'):return np.mean([vals(st,m,key,s,sel).mean() for s in (0,1)])
def object_means(st,m,key,sel='best_mse'):
    seeds=[0] if m=='BASE' else [0,1]
    by_seed=[{r['id']:float(r[key]) for r in group(st,m,seed,sel)} for seed in seeds]
    ids=sorted(by_seed[0])
    assert len(ids)==500 and all(set(v)==set(ids) for v in by_seed)
    return np.array([np.mean([v[oid] for v in by_seed]) for oid in ids])
def sci(x):
    if x==0:return '0'
    a,b=f'{x:.2e}'.split('e');return a+r'\times10^{'+str(int(b))+'}'
def table(name,headers,lines,spec):
    t=[r'\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}',r'\begin{tabular}{@{}'+spec+'@{}}',r'\toprule',' & '.join(headers)+r' \\ \midrule']
    t+=[' & '.join(line)+r' \\' if isinstance(line,list) else line for line in lines]
    t += [r'\bottomrule',r'\end{tabular}\endgroup']
    if name == 'budget_main':
        t = [line.replace(r'{\tabcolsep}{4pt}', r'{\tabcolsep}{14pt}').replace(' / ', r'\enspace/\enspace ') for line in t]
    (TAB/(name+'.tex')).write_text('\n'.join(t)+'\n')
# Check the actual data and training contracts before rendering.
splits=read(DATA/'metadata/split_ids.json');runs=read(DATA/'metadata/training_runs.json.xz');profile=read(DATA/'metadata/profile.json')
assert {k:len(v) for k,v in splits.items()}=={'train':2500,'validation':500,'test':500}
assert len(set(sum(splits.values(),[])))==3500
assert len(runs)==16 and len(G)==34 and len(R)==17000
assert all(len(g)==500 and set(r['id'] for r in g)==set(splits['test']) for g in G.values())
for run in runs:
    assert run['summary']['position']['epoch']==60 and run['summary']['position']['step']==75000
    assert run['summary']['status']=='completed' and run['summary']['test_read_count']==0
    for split in ('train','validation'):assert set(run['configuration']['dataset']['used_ids'][split])==set(splits[split])
assert profile['epochs']==60 and profile['beta_multipliers']==[1.0]
for st,expected in [('tikhonov',[2,2]),('tv',[258,292])]:
    assert [sum(r['clipping_active'] is True for r in group(st,'DP',seed)) for seed in (0,1)]==expected
    assert all(sum(r['clipping_active'] is True for r in group(st,'DP_TWO',seed))==500 for seed in (0,1))
assert all(isinstance(r['constraint_violation'],bool) for r in R if r['method'] in ('DP','DP_TWO','NSN'))
assert not any(r['constraint_violation'] for r in R if r['method'] in ('DP','DP_TWO','NSN'))
for run in runs:
    c=run['configuration']
    assert c['training']['batch_size']==2 and c['training']['gradient_clip_norm']==1
    assert c['optimizer']['lr']==1e-4 and c['optimizer']['weight_decay']==0
    assert c['scheduler']['step_size']==10 and c['scheduler']['gamma']==.5
    assert c['computation']['dtype']=='float64'
    assert c['geometry']['size']==256 and c['geometry']['angles']==120 and c['geometry']['detectors']==363
    assert len(run['segments'])==2 and not c['selection']['early_stopping_enabled']
    assert run['projection_contract']['basis_shape']==[65536,38597]

# Main quality: object-wise seed means, then sample SD across 500 objects.
lines=[];object_spread={}
for st in starts:
    lines.append(r'\multicolumn{6}{@{}l}{\textit{'+ST[st]+r' initialization}} \\')
    object_spread[st]={}
    for m in M:
        quality=[];object_spread[st][m]={}
        for key,scale,digits in [('mse',1e4,3),('psnr',1,2),('ssim',1,4)]:
            v=object_means(st,m,key);object_spread[st][m][key]={'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1))}
            assert np.isclose(v.mean(),mean(st,m,key),rtol=1e-13)
            quality.append(f'${scale*v.mean():.{digits}f}\\pm{scale*v.std(ddof=1):.{digits}f}$')
        d=mean(st,m,'data_change_over_noise')
        lines.append([LABEL[m]]+quality+[('$'+sci(d)+'$') if 0<d<1e-5 else f'{d:.3f}',f"{mean(st,m,'residual_over_noise'):.3f}"])
    if st==starts[0]:lines.append(r'\midrule')
table('quality_main',['Method',r'MSE $[10^{-4}]$','PSNR [dB]','SSIM',r'$d/\delta$',r'$r/\delta$'],lines,'lrrrrr')
# Main control table: both seeds in a single cell (never pooled trials).
lines=[]
for st in starts:
    lines.append(r'\multicolumn{4}{@{}l}{\textit{'+ST[st]+r' initialization}} \\')
    for m in M[1:]:
        gg=[group(st,m,s) for s in (0,1)]
        clipped=' / '.join(str(sum(r['clipping_active'] is True for r in g)) for g in gg) if m in ('DP','DP_TWO') else '--'
        maximum=' / '.join(('$'+sci(max(r['data_change_over_noise'] for r in g))+'$') if m=='NSN' else f"{max(r['data_change_over_noise'] for r in g):.3f}" for g in gg)
        above=' / '.join(str(sum(r['reference_budget_excess']>0 for r in g)) for g in gg) if m=='RES' else '--'
        lines.append([LABEL[m],clipped,maximum,above])
    if st==starts[0]:lines.append(r'\midrule')
table('budget_main',['Method',r'Clipped',r'Max. $d/\delta$',r'RES: $d>\delta$'],lines,'lrrr')
# Complete seed-level quality and numerical diagnostics, both selection rules.
for sel in ('best_mse','best_ssim'):
    q=[];b=[];dc=[]
    for st in starts:
        for m in M:
            for s in ([0] if m=='BASE' else [0,1]):
                g=group(st,m,s,sel); name=[ST[st],LABEL[m], '--' if m=='BASE' else str(s)]
                fmt=[]
                for k,scale,dig in [('mse',1e4,3),('psnr',1,2),('ssim',1,4)]:
                    v=vals(st,m,k,s,sel)*scale
                    fmt.append(f'${v.mean():.{dig}f}\\pm{v.std(ddof=1):.{dig}f}$')
                q.append(name+fmt+[f"{vals(st,m,'data_change_over_noise',s,sel).mean():.3g}",f"{vals(st,m,'residual_over_noise',s,sel).mean():.3f}"])
                dc.append(name+['$'+sci(vals(st,m,k,s,sel).mean())+'$' for k in ('error_null_mse','error_range_mse')]+['$'+sci(max(abs(r['error_cross_term']) for r in g))+'$'])
                if m in ('NSN','DP','DP_TWO'):
                    b.append(name+['$'+sci(max(r['null_leak']/r['delta_noise'] for r in g))+'$', '$'+sci(max(r['signed_beta_excess'] for r in g))+'$', '$'+sci(max(r['signed_bound_excess'] for r in g))+'$',str(sum(r['constraint_violation'] is True for r in g))])
    table('quality_'+sel,['Start','Method','Seed',r'MSE $[10^{-4}]$','PSNR [dB]','SSIM',r'$d/\delta$',r'$r/\delta$'],q,'llcrrrrr')
    table('bounds_'+sel,['Start','Method','Seed',r'Max. $\ell/\delta$',r'Max. $d-\beta$',r'Max. $d-B$','Viol.'],b,'llcrrrr')
    table('decomp_'+sel,['Start','Method','Seed',r'Mean $E_N$',r'Mean $E_R$',r'Max. $|C|$'],dc,'llcrrr')
    pp=[]
    for p in read(DATA/'test/paired_effects.json'):
        if p['selection']!=sel or p['metric']!='mse':continue
        lo,hi=p['bca']['interval'];ss='Mean' if p['seed_index']=='mean_over_seeds' else str(p['seed_index'])
        pp.append([ST[p['start']],LABEL[p['left']]+' -- '+LABEL[p['right']],ss,'$'+sci(p['summary']['mean'])+'$', '$['+sci(lo)+', '+sci(hi)+']$',str(round(500*p['worsened_fraction']))])
    table('paired_'+sel,['Start','Difference','Seed',r'$\Delta$MSE',r'95\% BCa interval','Worse / 500'],pp,'llcrrr')
# Observational computational costs from the delivered, hash-bound comparison.
COST=args.cost_file.resolve()
cost=read(COST)
for rel in ['data/metadata/training_runs.json.xz','data/test/individual_values.csv.gz','data/test/paired_effects.json']:
    reference=next(v for k,v in cost['input_references'].items() if k == rel or k.endswith('/'+rel))
    assert hashlib.sha256((SRC/rel).read_bytes()).hexdigest()==reference['sha256']
for cr in cost['training_runs']:
    run=next(r for r in runs if r['id']==cr['job'])
    assert np.isclose(cr['numerical_hours']*3600,run['resource_summary']['numerical_seconds_sum'])
    assert cr['epochs']==60 and cr['steps']==75000
cost_summary={}
for m in ['DP','DP_TWO']:
    tr=[r for r in cost['training_runs'] if r['method']==m]
    tg=[r for r in cost['test_groups'] if r['method']==m and r['selection']=='best_mse' and r['seed_index'] in ['0','1']]
    assert len(tr)==len(tg)==4
    for t in tg:
        g=group(t['start'],m,int(t['seed_index']))
        assert {r['parameter_count'] for r in g}=={t['parameter_count']}
        assert np.isclose(np.mean([r['inference_seconds'] for r in g]),t['inference_seconds_mean'],rtol=1e-13)
    cost_summary[m]={'parameters':tg[0]['parameter_count'],
                     **{k:float(np.mean([r[k] for r in tr])) for k in ['numerical_hours','segment_wall_hours','weights_mib']},
                     'inference_ms':float(1000*np.mean([r['inference_seconds_mean'] for r in tg]))}
ct=[]
for label,k,digits in [('Trainable parameters','parameters',0),('Logged optimization time [h]','numerical_hours',2),('Training segment elapsed time [h]','segment_wall_hours',2),('Forward pass, batch 1 [ms]','inference_ms',1),('Saved model weights [MiB]','weights_mib',2)]:
    v=[cost_summary[m][k] for m in ['DP','DP_TWO']]
    ct.append([label]+[f'{x:,.0f}' if digits==0 else f'{x:.{digits}f}' for x in v]+[f'{v[1]/v[0]:.2f}'])
table('cost_summary',['Quantity','DP1','DP2','DP2 / DP1'],ct,'lrrr')
ct=[]
for r in sorted(cost['training_runs'],key=lambda r:(r['start'],r['seed_index'],r['method'])):
    g=group(r['start'],r['method'],r['seed_index'])
    ct.append([ST[r['start']],LABEL[r['method']],str(r['seed_index']),f"{r['numerical_hours']:.3f}",f"{r['segment_wall_hours']:.3f}",f"{1000*np.mean([v['inference_seconds'] for v in g]):.2f}"])
table('cost_runs',['Start','Method','Seed','Optimization [h]','Elapsed [h]','Forward [ms]'],ct,'llcrrr')
revision={'operation':'CPU aggregation of existing test metrics and reported timings; no training/inference',
          'cost_source':str(COST),'cost_source_sha256':hashlib.sha256(COST.read_bytes()).hexdigest(),
          'cost_summary':cost_summary,'timing_contract':cost['timing_contract'],
          'main_quality_object_spread':object_spread,
          'sd_definition':'Sample SD (ddof=1) across 500 object-wise seed-averaged metrics, not across seeds or reconstructed image ensembles',
          'main_example':{'id':'000002','seed':0,'selection':'best_mse','selection_reason':'Spatial detail among already exported prespecified diagnostic objects; chosen after inspection of targets, not by reconstruction performance','intensity_window':[0,.4],'error_window':[0,.25]}}
(OUTPUT/'cost-and-dispersion-audit.json').write_text(json.dumps(revision,indent=2)+'\n')
if args.tables_only:
    print('Regenerated tables with object SDs, no bold values, and verified cost summaries.')
    sys.exit(0)
# Paper-sized vector plots and original-array reconstruction panels.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors={'RES':'#0072B2','DP':'#009E73','DP_TWO':'#CC79A7'}
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None});fig.savefig(PREVIEW/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(6.0,2.45),layout='constrained')
for ax,st in zip(axes,starts):
    for m in colors:
        for s in (0,1):
            v=np.sort(vals(st,m,'data_change_over_noise',s));v=np.round(v,12) if m=='DP_TWO' else v
            ax.step(np.r_[0,v,1.2 if st=='tikhonov' else 10],np.r_[0,np.arange(1,501)/500,1],where='post',color=colors[m],ls='-' if s==0 else '--',lw=1.2,label=LABEL[m] if s==0 else None)
    ax.axvline(1,color='#333333',ls=':',lw=1);ax.set(title=ST[st],xlabel=r'Additional data change $d/\delta$',ylabel='Fraction of test objects',ylim=(0,1.02),xlim=(0,1.2 if st=='tikhonov' else 10));ax.grid(alpha=.2);ax.legend(loc='lower right')
    if st=='tv':
        ia=ax.inset_axes([.48,.32,.47,.45]);
        for m in colors:
            for s in (0,1):
                v=np.sort(vals(st,m,'data_change_over_noise',s));v=np.round(v,12) if m=='DP_TWO' else v
                ia.step(np.r_[0,v,10],np.r_[0,np.arange(1,501)/500,1],where='post',color=colors[m],ls='-' if s==0 else '--',lw=.8)
        ia.axvline(1,color='#333333',ls=':',lw=.8);ia.set(xlim=(.5,1.2),ylim=(0,1.02));ia.tick_params(labelsize=6);ia.set_title('Budget region',fontsize=6)
save(fig,'data_change_ecdf')
fig,axes=plt.subplots(1,2,figsize=(6,2.6),layout='constrained')
for ax,st in zip(axes,starts):
    x=np.arange(5)
    for j,(k,label,col) in enumerate([('error_null_mse',r'$E_N$','#56779B'),('error_range_mse',r'$E_R$','#D59B45')]):
        values=[1e4*mean(st,m,k) for m in M];pos=x+(j-.5)*.34
        ax.bar(pos,values,.32,label=label,color=col)
        for s,marker in [(0,'o'),(1,'x')]:ax.scatter(pos,[1e4*vals(st,m,k,s).mean() for m in M],s=9,color='black',marker=marker,linewidths=.5,zorder=3)
    ax.set(title=ST[st],xticks=x,xticklabels=[LABEL[m] for m in M],ylabel=r'Component MSE $[10^{-4}]$');ax.grid(axis='y',alpha=.2);ax.legend()
save(fig,'error_decomposition')
index=read(DATA/'examples/index.json')
def reconstruction(oid,seed,sel,name,intensity_max=1.0,error_max=.6):
    fig,ax=plt.subplots(6,6,figsize=(7.0,6.7),gridspec_kw={'hspace':.04,'wspace':.04})
    fig.subplots_adjust(left=.10,right=.93,bottom=.055,top=.95)
    with np.load(DATA/f'examples/target_{oid}.npz') as z:target=z['target'].squeeze()
    titles=['Ground truth']+[LABEL[m] for m in M]
    for b,st in enumerate(starts):
        images=[target]
        for m in M:
            candidates=[p for p in index['predictions'] if p['id'].endswith('/'+oid) and p['start_kind']==st and p['method']==m and (m=='BASE' or (p['seed_index']==seed and p['selection']==sel))]
            assert candidates,(oid,st,m,seed,sel)
            p=candidates[0]
            with np.load(SRC/p['file']) as z:im=z['prediction'].squeeze();images.append(im)
            g=group(st,m,seed,sel);rr=next(r for r in g if r['id'].endswith('/'+oid))
            assert np.isclose(np.mean((im-target)**2),rr['mse'],rtol=1e-10,atol=1e-15)
        for c,im in enumerate(images):
            ax[3*b,c].imshow(im,cmap='gray',vmin=0,vmax=intensity_max,origin='lower');ax[3*b,c].add_patch(Rectangle((64,64),128,128,fill=False,edgecolor='#00BFFF',lw=.5))
            err=ax[3*b+1,c].imshow(abs(im-target),cmap='magma',vmin=0,vmax=error_max,origin='lower')
            ax[3*b+2,c].imshow(im[64:192,64:192],cmap='gray',vmin=0,vmax=intensity_max,origin='lower')
            if b==0:ax[0,c].set_title(titles[c],fontsize=8)
        for k,label in enumerate([ST[st]+'\nreconstruction','Absolute error','Central zoom']):ax[3*b+k,0].set_ylabel(label,fontsize=8)
    for a in ax.flat:a.set(xticks=[],yticks=[]);a.spines[['top','bottom','left','right']].set_visible(False)
    ca=fig.add_axes([.945,.36,.012,.27]);fig.colorbar(err,cax=ca,ticks=[0,error_max/2,error_max]);ca.tick_params(labelsize=7)
    save(fig,name)
reconstruction('000002',0,'best_mse','reconstructions_main',intensity_max=.4,error_max=.25)
for oid,s,sel,name in [('000001',0,'best_mse','example_1'),('000002',0,'best_mse','example_2'),('000000',1,'best_mse','example_seed1'),('000000',0,'best_ssim','example_ssim0'),('000000',1,'best_ssim','example_ssim1'),('000490',0,'best_mse','example_difficult')]:reconstruction(oid,s,sel,name)
# Training curves and selection epochs, all 16 runs.
curves=rows(DATA/'validation_curves.csv');epochs=rows(DATA/'training_epochs.csv');inv=[]
for run in sorted(runs,key=lambda r:(r['start'],r['seed_index'],M.index(r['method']))):
    inv.append([ST[run['start']],LABEL[run['method']],str(run['seed_index']),str(run['best_mse']['epoch']+1),str(run['best_ssim']['epoch']+1),'60','75,000'])
table('checkpoints',['Start','Method','Seed','MSE epoch','SSIM epoch','Epochs','Steps'],inv,'llcrrrr')
for st in starts:
    fig,axes=plt.subplots(2,2,figsize=(6,4.1),layout='constrained')
    for seed in (0,1):
        for m in M[1:]:
            run=next(r for r in runs if r['start']==st and r['seed_index']==seed and r['method']==m)
            c=sorted([r for r in curves if r['job']==run['id']],key=lambda r:float(r['epoch']))
            assert len(c)==60
            for j,key in enumerate(['mse','ssim']):axes[j,seed].plot([float(r['epoch']) for r in c],[float(r[key]) for r in c],label=LABEL[m],lw=1)
        for j,key in enumerate(['MSE','SSIM']):
            axes[j,seed].set(title=f'{ST[st]}, seed {seed}',xlabel='Epoch',ylabel='Validation '+key);axes[j,seed].axvline(30,color='gray',ls=':',lw=.8);axes[j,seed].grid(alpha=.2)
        axes[0,seed].set_yscale('log');axes[0,seed].legend(ncol=2)
    save(fig,'validation_'+st)
# Machine-readable trace of the transformation and numeric claims.
claims={st:{m:{k:float(mean(st,m,k)) for k in ['mse','psnr','ssim','data_change_over_noise','residual_over_noise','error_null_mse','error_range_mse']} for m in M} for st in starts}
checks={'split_counts':{k:len(v) for k,v in splits.items()},'training_runs':len(runs),'epochs_per_run':60,'steps_per_run':75000,'test_groups':len(G),'test_rows':len(R),'groups_complete':True,'input_sha256':{str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'test/individual_values.csv.gz',DATA/'metadata/training_runs.json.xz',DATA/'metadata/split_ids.json',DATA/'test/paired_effects.json',DATA/'examples/index.json']},'claims':claims,'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'operation':'CPU aggregation and plotting only; no new training/inference/bootstrap'}
audit_path=OUTPUT/'generation-audit.json'
previous=read(audit_path) if audit_path.exists() else {}
previous.update(checks)
audit_path.write_text(json.dumps(previous,indent=2)+'\n')
print(json.dumps(claims,indent=2));print('All artifact generation checks passed.')
