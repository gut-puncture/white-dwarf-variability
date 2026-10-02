# The observations belong to the survey teams

This work uses public observations from **ZTF/IPAC, TESS/NASA/MAST, SDSS, and ESA Gaia**, with catalogue checks through CDS SIMBAD/VizieR and AAVSO VSX. Credit those teams when using the data or results. The repository's MIT licence covers its analysis code, not a new licence on third-party observations.

The [raw manifest](../provenance/raw_manifest.json) lists every included archive file, its size, SHA-256 checksum, and original URL or query. The release contains the downloaded observations, not published paper PDFs. Archive cutouts can be regenerated with different headers or later calibrations; the versioned release preserves the bytes analysed here.

## Data used in the reproduction

| Source | What we use | Documentation |
| --- | --- | --- |
| ZTF, hosted by IPAC/IRSA | Original g/r light-curve CSVs for both stars; difference images, masks and PSFs for the second | [Archive API](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html), [Masci et al. data-system paper](https://arxiv.org/abs/1902.01872) |
| TESS, hosted by MAST | Sector 72 SPOC light curve and target pixels; Tesscut images in sectors 22, 45, 46 and 49 | [TESS archive](https://archive.stsci.edu/missions-and-data/tess), [data products](https://archive.stsci.edu/missions-and-data/tess/data-products), [MAST attribution guidance](https://archive.stsci.edu/publishing/data-attributions) |
| SDSS DR20 | 25 target visit files and 125 comparison-star visit files, including individual exposure spectra | [DR20](https://www.sdss.org/dr20/), [data access](https://www.sdss.org/dr20/data_access/get_data/), [SDSS acknowledgements](https://www.sdss.org/collaboration/) |
| Gaia DR3, ESA/DPAC | Positions, proper motions and neighbouring-source catalogue; parent-sample quality fields | [Gaia Archive](https://gea.esac.esa.int/archive/) |

The 400-star selection came from the SDSS DR20 SnowWhite product `astraAllStarSnowWhite-0.8.1.fits`, combined with Gaia fields. The selected catalogue rows and screening outputs are retained in `provenance/`; the large parent FITS file and unrelated stars' observations are outside this release.

## Previously published work

- [Kepler et al. (2016)](https://academic.oup.com/mnras/article/455/4/3413/1264189): the first star's magnetic spectrum, under SDSS J112148.77+103934.1.
- [Heinze et al. (2018), ATLAS catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/156/241): the first star's earlier dubious-variable flag, ATO J170.4531+10.6594.
- [Amorim et al. (2023)](https://arxiv.org/abs/2301.08862): magnetic-white-dwarf catalogue and model-dependent field estimate.
- [Hardy, Dufour & Jordan (2023)](https://arxiv.org/abs/2301.06596), [companion study](https://arxiv.org/abs/2301.06605): magnetic-atmosphere fits and classification difficulties.
- [Gentile Fusillo et al. (2021) catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/508/3877): previously published, atmosphere-dependent white-dwarf properties. This work does not discover or independently measure those masses.

Further period-catalogue comparisons are recorded in [the novelty audit](NOVELTY.md). A journal manuscript should include the surveys' requested full acknowledgements and relevant software citations in addition to these links.

## Authorship and review

Shailesh Rana directed the research and maintains this repository. OpenAI Codex performed the analysis and wrote the code and documentation. No independent astronomer has yet reviewed the result. The repository is a public research record, not a peer-reviewed paper.
