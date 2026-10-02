"""Pixel localization and whole-block nulls for the newly tested TESS sectors."""
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=importlib.import_module('61247086_tesscut_validation');ROOT=base.ROOT;SID=80364427;SECTORS=[22,45,46,49]

def design(t,y,error,frozen,tref,exptime):
 B=base.make_nuisance(t);u,s,_=np.linalg.svd(B/error[:,None],full_matrices=False);rank=int(sum(s>s[0]*1e-10));Q=u[:,:rank]
 project=lambda x:x-Q@(Q.T@x);wy=project(y/error);scale=max(1,float(wy@wy)/(len(t)-rank));wy/=np.sqrt(scale);bases=[]
 for candidate in frozen:
  f=candidate['frequency'];theta=2*np.pi*f*(t-tref);X=project(np.sinc(f*exptime/86400)*np.c_[np.sin(theta),np.cos(theta)]/error[:,None]);u,s,_=np.linalg.svd(X,full_matrices=False);assert s[-1]>1e-8*s[0];bases.append(u[:,:2])
 return wy,Q,scale,rank,np.hstack(bases)

def aggregate(matrix,y,groups):
 a=np.zeros((matrix.shape[1],int(max(groups)+1)))
 for k in range(a.shape[1]):m=groups==k;a[:,k]=matrix[m].T@y[m]
 return a

def main():
 fz=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());f=fz['selected'][0]['frequency'];tref=fz['tref'];metadata=json.loads((ROOT/f'data/{SID}_additional_tess_validation.json').read_text())['metadata'];cache=[];pixels=[];tests=[]
 fig,axs=plt.subplots(4,2,figsize=(10,14))
 for k,sector in enumerate(SECTORS):
  d=pd.read_csv(ROOT/f'data/{SID}_tesscut_s{sector}_photometry.csv.gz');cube=np.load(ROOT/f'data/{SID}_tesscut_s{sector}_pixels.npz');meta=next(r for r in metadata if r['sector']==sector)
  t=d.time.to_numpy();y=d.flux_r150.to_numpy();e=d.error_r150.to_numpy();wy,Q,scale,rank,basis=design(t,y,e,fz['selected'],tref,meta['exptime']);observed=np.sum((basis.T@wy).reshape(20,2)**2,axis=1)
  cache.append(dict(sector=sector,t=t,y=wy,Q=Q,scale=scale,rank=rank,basis=basis,observed=observed))
  flux=cube['flux'];error=cube['error'];B=base.make_nuisance(t);records=[]
  for j in range(flux.shape[1]):
   r=base.fit(t,flux[:,j],error[:,j],f,tref,meta['exptime'],B);r['pixel_index']=j;r['pixel_y'],r['pixel_x']=divmod(j,int(cube['shape'][1]));records.append(r)
  best=max(records,key=lambda r:r['delta_chi2']);x0,y0=meta['target_xy'];best['distance_to_target_pixels']=float(np.hypot(best['pixel_x']-x0,best['pixel_y']-y0));pixels.append(dict(sector=sector,target_xy=[x0,y0],maximum=best,all_pixels=records))
  delta=np.array([r['delta_chi2'] for r in records]).reshape(cube['shape']);axs[k,0].imshow(np.log10(np.maximum(cube['reference'],1e-3)),origin='lower',cmap='magma');im=axs[k,1].imshow(delta,origin='lower',cmap='viridis');fig.colorbar(im,ax=axs[k,1],label='Fixed-period improvement')
  for ax in axs[k]:ax.plot(x0,y0,'+',color='red',ms=12);ax.set(xlabel='Pixel x',ylabel='Pixel y')
  axs[k,0].set_title(f'Sector{sector}: mean image');axs[k,1].set_title(f'117.406-minute modulation; max offset{best["distance_to_target_pixels"]:.2f}px')
  print('PIXEL',sector,{name:best[name] for name in ['pixel_x','pixel_y','delta_chi2','distance_to_target_pixels']},flush=True)
 for width in [.2,.5,1.]:
  rng=np.random.default_rng(80364427);prepared=[]
  for s in cache:
   _,groups=np.unique(np.floor((s['t']-s['t'].min())/width),return_inverse=True);prepared.append(dict(**s,groups=groups,A=aggregate(s['basis'],s['y'],groups),N=aggregate(s['Q'],s['y'],groups)))
  per_sector={s['sector']:dict(exceedances=0,max_simulated=-np.inf) for s in prepared};combined_exceed=0;combined_max=-np.inf;combined_observed=float(sum(s['observed'][0] for s in prepared))
  for start in range(0,100000,1000):
   summed=np.zeros((20,1000))
   for s in prepared:
    signs=rng.choice([-1.,1.],size=(s['A'].shape[1],1000));values=s['A']@signs;nuis=s['N']@signs
    projected_norm=np.sum(s['y']**2)-np.sum(nuis**2,axis=0);ratio=np.maximum(1,projected_norm*s['scale']/(len(s['y'])-s['rank']))/s['scale']
    gain=np.sum(values.reshape(20,2,1000)**2,axis=1)/ratio;summed+=gain;maximum=gain.max(axis=0);p=per_sector[s['sector']];p['exceedances']+=int(sum(maximum>=s['observed'][0]));p['max_simulated']=max(p['max_simulated'],float(max(maximum)))
   maximum=summed.max(axis=0);combined_exceed+=int(sum(maximum>=combined_observed));combined_max=max(combined_max,float(max(maximum)))
  for s in prepared:
   p=per_sector[s['sector']];p.update(sector=s['sector'],blocks=int(s['A'].shape[1]),observed=float(s['observed'][0]),plus_one_tail=(p['exceedances']+1)/100001)
  r=dict(block_days=width,draws=100000,per_sector=list(per_sector.values()),combined=dict(observed=combined_observed,exceedances=combined_exceed,max_simulated=combined_max,plus_one_tail=(combined_exceed+1)/100001,
   statistic='Sum of improvements at the same frozen frequency across all four independent sectors; null max over20original frequencies'))
  tests.append(r);print('BLOCK NULL',r,flush=True)
  (ROOT/f'data/{SID}_additional_tess_audit.json').write_text(json.dumps(dict(pixel_checks=pixels,block_nulls=tests,limitations=[
   'Faint-sky plane induces pixel covariance. Pixel maps are localization diagnostics; their pointwise statistics are not discovery probabilities.',
   'Block signs preserve observed residual shapes within a block and assume symmetric independent blocks; different block sizes share photons.',
   'Original20frequencies are held fixed. The exploratory secondary-clock test is separate and produced no convincing replication.',
   'No TESS flux clipping, no phase optimization, no sector removal. Error inflation is refitted under each block-sign nuisance projection.',
   'Finite100000draws bound the null resolution; zero exceedances do not mean zero probability.']),indent=2)+'\n')
 fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_additional_tess_pixels.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
