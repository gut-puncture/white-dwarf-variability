"""Independent TESS test at frequencies frozen from ZTF discovery data."""
from pathlib import Path
import os,json,argparse
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
from scipy.stats import chi2
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def fit(t,y,e,f,tref,width=.2,exposure_seconds=120):
    group=np.floor(t/width).astype(int);_,ix=np.unique(group,return_inverse=True);w=1/e**2;weight=np.bincount(ix,w)
    def project(v):return v-(np.bincount(ix,w*v)/weight)[ix]
    yy=project(y);null=float(np.sum(w*yy**2));dof=max(1,len(y)-len(weight));scale=max(1.,null/dof)
    phi=2*np.pi*f*(t-tref);response=np.sinc(f*exposure_seconds/86400)
    z=np.column_stack([project(np.sin(phi)*response),project(np.cos(phi)*response)])
    gram=z.T@(w[:,None]*z);cov=np.linalg.inv(gram)*scale;coef=np.linalg.solve(gram,z.T@(w*yy));gain=float(coef@gram@coef)/scale
    amp=float(np.hypot(*coef));grad=np.array([-coef[1],coef[0]])/max(amp**2,1e-30)
    a_grad=coef/max(amp,1e-30)
    return dict(frequency=f,period_seconds=86400/f,delta_chi2=gain,unscaled_delta_chi2=gain*scale,
        variance_inflation=scale,amplitude=amp,amplitude_error=float(np.sqrt(a_grad@cov@a_grad)),
        fractional_amplitude=amp/np.median(y),phase_radians=float(np.arctan2(coef[1],coef[0])),phase_error_radians=float(np.sqrt(grad@cov@grad)),
        coefficients=coef.tolist(),covariance=cov.tolist(),nominal_tail=float(chi2.sf(gain,2))),dict(residual=yy,error=e*np.sqrt(scale),phase=(f*(t-tref))%1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sid',type=int,required=True);parser.add_argument('--cadence',type=int,choices=[20,120],default=120);parser.add_argument('--preferred-tic',type=int);a=parser.parse_args();sid=a.sid
    tag='' if a.cadence==120 else '_20sec'
    frozen=json.loads((ROOT/f'data/fast_rotation/{sid}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];frequencies=[r['frequency'] for r in frozen['selected']]
    rows=[];meta=[];plotdata=[];by_sector={};excluded=[]
    for file in sorted((ROOT/f'raw/tess/{sid}').glob('*s_lc.fits' if a.cadence==120 else '*a_fast-lc.fits')):
        with fits.open(file) as h:
            by_sector.setdefault(int(h[0].header['SECTOR']),[]).append((int(h[0].header['TICID']),file))
    selected_files=[]
    for sector,items in sorted(by_sector.items()):
        if len(items)>1:
            preferred=[p for tic,p in items if tic==a.preferred_tic]
            assert len(preferred)==1, f'Duplicate sector {sector}: explicitly choose a verified TIC identity with --preferred-tic'
            selected_files.extend(preferred)
            excluded.extend(dict(sector=sector,ticid=tic,file=str(p.relative_to(ROOT)),reason='Duplicate photons under another TIC entry') for tic,p in items if p not in preferred)
        else:selected_files.append(items[0][1])
    for file in selected_files:
        with fits.open(file) as h:
            data=h[1].data;mask=data['QUALITY']==0
            cols=['TIME','SAP_FLUX','SAP_FLUX_ERR','PDCSAP_FLUX','PDCSAP_FLUX_ERR']
            for col in cols:mask&=np.isfinite(data[col])
            mask&=(data['SAP_FLUX_ERR']>0)&(data['PDCSAP_FLUX_ERR']>0)
            d=pd.DataFrame({col:np.asarray(data[col][mask],float) for col in cols})
            sector=int(h[0].header['SECTOR']);header={key:h[1].header.get(key) for key in ['CROWDSAP','FLFRCSAP','TIMEDEL','BJDREFI','BJDREFF','TIMESYS']}
            t=d.TIME.to_numpy()+float(h[1].header['BJDREFI'])+float(h[1].header.get('BJDREFF',0))-2458000
            d['time']=t
            meta.append(dict(sector=sector,ticid=int(h[0].header['TICID']),ra=float(h[0].header['RA_OBJ']),dec=float(h[0].header['DEC_OBJ']),file=str(file.relative_to(ROOT)),n_input=len(data),n_clean=len(d),header=header,
                median_pdc_error=float(np.median(d.PDCSAP_FLUX_ERR)),median_pdc_flux=float(np.median(d.PDCSAP_FLUX)),time_system='BJD TDB minus2458000'))
        d.to_csv(ROOT/f'data/{sid}_tess_s{sector}{tag}_clean.csv.gz',index=False)
        for rank,f in enumerate(frequencies,1):
            row=dict(sector=sector,discovery_rank=rank,frequency=f,period_seconds=86400/f)
            for label,prefix in [('pdc','PDCSAP'),('sap','SAP')]:
                result,detail=fit(t,d[prefix+'_FLUX'].to_numpy(),d[prefix+'_FLUX_ERR'].to_numpy(),f,tref,exposure_seconds=a.cadence)
                row[label]=result
                if rank==1 and label=='pdc':plotdata.append((sector,t,d,detail,result))
            rows.append(row)
    combined=[]
    for rank,f in enumerate(frequencies,1):
        selected=[r for r in rows if r['discovery_rank']==rank];gain=sum(r['pdc']['delta_chi2'] for r in selected)
        combined.append(dict(discovery_rank=rank,frequency=f,period_seconds=86400/f,independent_sector_gain=gain,
                             nominal_20_frequency_tail=min(1.,20*float(chi2.sf(gain,2*len(selected))))))
    output=dict(sdss_id=sid,cadence_seconds=a.cadence,selection='20 discovery-only ZTF frequencies; no TESS frequency refinement',tref=tref,sectors=meta,excluded_duplicate_products=excluded,preferred_tic=a.preferred_tic,tests=rows,combined=combined,
        caveats=['SAP and PDC share photons; do not combine them as independent evidence.',
        'Variance inflation is estimated under the no-periodic-signal model separately in each sector.',
        'Two sine/cosine parameters per sector; independent phases and amplitudes are allowed.',
        'A TESS signal still requires pixel source localization and alias checks.'])
    (ROOT/f'data/{sid}_fast_rotation_tess_validation{tag}.json').write_text(json.dumps(output,indent=2)+'\n')
    shown=plotdata[:3]
    fig,axs=plt.subplots(len(shown),2,figsize=(11,4*len(shown)),squeeze=False)
    for i,(sector,t,d,detail,result) in enumerate(shown):
        ax=axs[i,0];p=detail['phase'];y=detail['residual']/np.median(d.PDCSAP_FLUX);e=detail['error']/np.median(d.PDCSAP_FLUX);binid=np.floor(p*24).astype(int)
        ax.scatter(p,y,s=1,alpha=.06,color='#5379a3')
        for k in range(24):
            m=binid==k
            if m.sum():w=1/e[m]**2;ax.errorbar((k+.5)/24,np.average(y[m],weights=w),1/np.sqrt(w.sum()),fmt='o',ms=4,color='black')
        ax.set(xlabel='Phase at discovery-only ZTF frequency',ylabel='Fractional PDC flux residual',title=f'Sector {sector}: fixed-frequency Δχ²={result["delta_chi2"]:.1f}')
        ax.set_ylim(*np.percentile(y,[1,99]))
        halfwidth=max(3,.055*frequencies[0]);frequency=np.linspace(max(.01,frequencies[0]-halfwidth),frequencies[0]+halfwidth,15000);ls=LombScargle(t,detail['residual'],detail['error']);power=ls.power(frequency,method='fast')
        axs[i,1].plot(frequency,power,lw=.7);axs[i,1].axvline(frequencies[0],color='red',ls='--',label='Frozen ZTF maximum')
        axs[i,1].set(xlabel='Frequency (cycles/day)',ylabel='LS power',title='Local periodogram for alias diagnosis');axs[i,1].legend(fontsize=8)
    fig.suptitle(f'{sid}: TESS{a.cadence}-second checks; first{len(shown)} of{len(plotdata)} sectors; source attribution pending');fig.tight_layout();fig.savefig(ROOT/f'figures/{sid}_fast_rotation_tess{tag}.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(sectors=meta,combined=sorted(combined,key=lambda r:-r['independent_sector_gain'])[:8]),indent=2),flush=True)

if __name__=='__main__':main()
