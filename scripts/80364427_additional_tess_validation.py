"""Test frozen primary and residual ZTF frequency lists in four new TESS sectors."""
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=importlib.import_module('61247086_tesscut_validation');ROOT=base.ROOT;SID=80364427
base.SID=SID;base.TARGET=SkyCoord(170.45311084*u.deg,10.65945823*u.deg);base.NEIGHBOUR=None
SECTORS=[22,45,46,49]

def main():
 original=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());secondary=json.loads((ROOT/f'data/fast_rotation_secondary/{SID}_frozen_candidates.json').read_text());tref=original['tref'];primary=original['selected'][0]['frequency']
 manifest=json.loads((ROOT/f'data/{SID}_additional_tesscut_acquisition.json').read_text());assert {r['sector'] for r in manifest if r['status']=='verified'}==set(SECTORS)
 results=[];metadata=[];fig,axs=plt.subplots(4,2,figsize=(11,13))
 for index,sector in enumerate(SECTORS):
  d,meta,reference=base.extract(sector);metadata.append(meta);t=d.time.to_numpy();theta=2*np.pi*primary*(t-tref)
  primary_columns=np.array([v for h in [1,2] for v in [np.sin(h*theta),np.cos(h*theta)]]).T
  for width in [1.,.5]:
   nuisance=base.make_nuisance(t,width)
   for radius in [1.,1.5,2.]:
    name=f'r{int(radius*100)}';y=d['flux_'+name].to_numpy();error=d['error_'+name].to_numpy()
    for family,selected,B in [('original',original['selected'],nuisance),('secondary_after_primary',secondary['selected'],np.c_[nuisance,primary_columns])]:
     for rank,candidate in enumerate(selected,1):
      f=candidate['frequency'];r=base.fit(t,y,error,f,tref,meta['exptime'],B);r.pop('nominal_60_frequency_sector_tail')
      r.update(sector=sector,frequency_family=family,discovery_rank=rank,frequency=f,period_seconds=86400/f,aperture_radius=radius,nuisance_segment_days=width,exposure_sinc=float(np.sinc(f*meta['exptime']/86400)),
       nominal_160_frequency_sector_tail=float(min(1.,160*chi2.sf(max(0,r['delta_chi2']),2))));results.append(r)
  chosen=next(r for r in results if r['sector']==sector and r['frequency_family']=='original' and r['discovery_rank']==1 and r['aperture_radius']==1.5 and r['nuisance_segment_days']==1)
  print('PRIMARY',chosen,flush=True)
  second=sorted([r for r in results if r['sector']==sector and r['frequency_family']=='secondary_after_primary' and r['aperture_radius']==1.5 and r['nuisance_segment_days']==1],key=lambda r:r['delta_chi2'],reverse=True);print('SECONDARY TOP',second[:3],flush=True)
  ax=axs[index,0];ax.imshow(np.log10(np.maximum(reference,1e-3)),origin='lower',cmap='magma');ax.plot(*meta['target_xy'],'+',color='cyan',ms=12);ax.set(title=f'Sector{sector}; target marked',xlabel='Pixel x',ylabel='Pixel y')
  B=base.make_nuisance(t);y=d.flux_r150.to_numpy();e=d.error_r150.to_numpy();b=np.linalg.lstsq(B/e[:,None],y/e,rcond=None)[0];res=y-B@b;phase=((t-tref)*primary)%1;ax=axs[index,1]
  for k in range(20):
   m=(phase>=k/20)&(phase<(k+1)/20)
   if sum(m):w=1/e[m]**2;ax.errorbar((k+.5)/20,np.average(res[m],weights=w),np.sqrt(chosen['variance_inflation']/sum(w)),fmt='o',color='#37628e')
  ax.set(title=f'Frozen117.406-minute cycle; Δ={chosen["delta_chi2"]:.2f}',xlabel='Phase',ylabel='Aperture flux residual(e−/s)')
  output=dict(metadata=metadata,tests=results,limitations=[
   'Original20plus residual20frequencies in four independent sectors give160primary frequency/sector comparisons. Sensitivities share photons.',
   'Secondary frequencies were frozen before any new-sector extraction. Later original ZTF data were previously inspected; secondary search is exploratory.',
   'Primary radius1.5pixels and daily offsets/slopes were fixed before measurement. Other radii and segment widths are sensitivities.',
   'Primary and first harmonic are nuisance terms for secondary frequencies. No secondary search is performed within TESS.',
   'Exposure attenuation is recorded; highly attenuated periods cannot be excluded by a null.',
   'Positive aperture signals require pixel localization and correlated-noise checks before attribution.',
   'Sector72was deliberately excluded because its200and120second products share photons.'])
  (ROOT/f'data/{SID}_additional_tess_validation.json').write_text(json.dumps(output,indent=2)+'\n')
 fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_additional_tess_validation.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
