"""Sensitivity checks of the frozen 4100-A feature; no new selection or frequency fit."""
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from astropy.io import fits
from helium_line_strengths import REJECT_OR
reg=importlib.import_module('80364427_continuum_controls');ROOT=reg.ROOT;SID=80364427

VARIANTS={
 'baseline':dict(degree=1,mask=REJECT_OR,line_coverage=.9,side_coverage=.8,inflate=False,floor=.5,trend=False),
 'quadratic_continuum':dict(degree=2),
 'sideband_variance':dict(inflate=True),
 'one_angstrom_floor':dict(floor=1.),
 'strict_mask':dict(mask=REJECT_OR|sum(1<<b for b in [17,20,21,23])),
 'strict_coverage':dict(line_coverage=.95,side_coverage=.9),
 'within_visit_slope':dict(trend=True),
}

def measure(d,config):
 w=10**d['loglam'].astype(float);y=d['flux'].astype(float);iv=d['ivar'].astype(float)
 good=(iv>0)&np.isfinite(y)&(d['and_mask']==0)&((d['or_mask']&config['mask'])==0)
 line=(w>=4060)&(w<4140);side=((w>=4000)&(w<4040))|((w>=4160)&(w<4200))
 if sum(good&line)<config['line_coverage']*sum(line) or sum(good&side)<config['side_coverage']*sum(side):return None
 line&=good;side&=good;x=(w-4100)/200;degree=config['degree']
 X=np.vander(x[side],degree+1,increasing=True);XL=np.vander(x[line],degree+1,increasing=True)
 cov=np.linalg.inv(X.T@(iv[side,None]*X));coef=cov@(X.T@(iv[side]*y[side]));cont=XL@coef
 if np.any(cont<=0):return None
 rchi=float(np.sum((y[side]-X@coef)**2*iv[side])/(sum(side)-degree-1))
 if config['inflate']:cov*=max(1,rchi)
 dw=np.gradient(w)[line];value=np.sum(dw*(1-y[line]/cont));grad=np.sum((dw*y[line]/cont**2)[:,None]*XL,axis=0)
 error=np.sqrt(np.sum(dw**2/iv[line]/cont**2)+grad@cov@grad+config['floor']**2)
 return dict(value=float(value),error=float(error),sideband_rchi2=rchi)

def fit(d,f,tref,trend=False,coefficients=None):
 t=d.time.to_numpy();groups=d.file.to_numpy();e=d.error.to_numpy();y=d.value.to_numpy();cols=[]
 for g in np.unique(groups):
  m=groups==g;cols.append(m.astype(float))
  if trend and sum(m)>=3 and np.ptp(t[m])>0:cols.append(np.where(m,(t-np.mean(t[m]))/np.ptp(t[m]),0))
 N=np.asarray(cols).T/e[:,None];u,s,_=np.linalg.svd(N,full_matrices=False);rank=int(sum(s>s[0]*1e-10));Q=u[:,:rank]
 if len(y)-rank<4:raise ValueError('Too few residual degrees of freedom')
 project=lambda x:x-Q@(Q.T@x)
 wy=project(y/e);scale=max(1,float(wy@wy)/(len(y)-rank));wy/=np.sqrt(scale)
 phi=2*np.pi*f*(t-tref);att=np.sinc(f*d.exptime.to_numpy()/86400)
 X=project(att[:,None]*np.c_[np.sin(phi),np.cos(phi)]/e[:,None])/np.sqrt(scale)
 s=np.linalg.svd(X,compute_uv=False)
 if s[-1]<s[0]*1e-8:raise ValueError('Unidentifiable sinusoid')
 cov=np.linalg.inv(X.T@X);b=cov@(X.T@wy) if coefficients is None else np.asarray(coefficients)
 residual=wy-X@b
 return dict(n=len(y),n_nights=int(d.mjd.nunique()),nuisance_rank=rank,variance_inflation=scale,
  delta_chi2=float(wy@wy-residual@residual),coefficients=b.tolist(),covariance=cov.tolist(),
  amplitude_angstrom=float(np.linalg.norm(b)),phase_radians=float(np.arctan2(b[1],b[0])))

def main():
 meta=pd.read_csv(ROOT/f'data/{SID}_continuum_windows.csv');rows=[]
 for name,g in meta.groupby('file'):
  with fits.open(ROOT/f'raw/fast_rotation/{SID}_spectra'/name) as h:
   for r in g.itertuples():
    for variant,changes in VARIANTS.items():
     config={**VARIANTS['baseline'],**changes};a=measure(h[r.hdu].data,config)
     if a:rows.append(dict(file=r.file,hdu=r.hdu,mjd=r.mjd,time=r.time,exptime=r.exptime,split=r.split,variant=variant,**a))
 d=pd.DataFrame(rows);d.to_csv(ROOT/f'data/{SID}_feature_sensitivity_measurements.csv',index=False)
 frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());f=frozen['selected'][0]['frequency'];tref=frozen['tref'];tests=[]
 for variant,g in d.groupby('variant'):
  config={**VARIANTS['baseline'],**VARIANTS[variant]};result=dict(variant=variant,config=config)
  for label,part in [('training',g[g.split=='discovery_spectra']),('validation',g[g.split=='validation_spectra']),('full_exploratory',g)]:
   try:result[label]=fit(part,f,tref,config['trend'])
   except (ValueError,np.linalg.LinAlgError) as exc:result[label]=dict(status='unidentifiable',reason=str(exc),n=len(part))
  if 'coefficients' in result['training']:
   try:result['validation_unchanged_prediction']=fit(g[g.split=='validation_spectra'],f,tref,config['trend'],result['training']['coefficients'])
   except (ValueError,np.linalg.LinAlgError) as exc:result['validation_unchanged_prediction']=dict(status='unidentifiable',reason=str(exc))
  tests.append(result)
 baseline=d[d.variant=='baseline'];fields=[]
 for field,g in baseline.groupby(baseline.file.str.split('-').str[1]):
  fields.append(dict(field=field,**fit(g,f,tref)))
 loo=[]
 for night,g in baseline.groupby('mjd'):
  try:
   training=fit(baseline[baseline.mjd!=night],f,tref);held=fit(g,f,tref,coefficients=training['coefficients']);loo.append(dict(mjd=int(night),**held))
  except (ValueError,np.linalg.LinAlgError) as exc:loo.append(dict(mjd=int(night),n=len(g),status='unidentifiable',reason=str(exc)))
 out=dict(tests=tests,observing_fields=fields,leave_night_out_exploratory=loo,frequency=f,
  limitations=['All are post-selection sensitivity diagnostics, not independent discoveries or additional blind validations.',
  'Changing continuum order changes the physical observable; the broad feature need not have invariant equivalent width.',
  'Within-visit slopes absorb real long arcs as well as calibration drifts.',
  'Strict masks add BOSS bits17,20,21,23 to the original mask; no clipping by measured flux or phase.'])
 (ROOT/f'data/{SID}_feature_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
 print([(r['variant'],r['full_exploratory'].get('n'),r['full_exploratory'].get('delta_chi2'),r.get('validation_unchanged_prediction',{}).get('delta_chi2')) for r in tests],flush=True)
 print('FIELDS',fields,flush=True)

if __name__=='__main__':main()
