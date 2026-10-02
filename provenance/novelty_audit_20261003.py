"""Read-only, bounded catalogue audit for the two leading variability claims.

Empty valid query results are absence from the named catalogue only. They are
never proof of universal novelty. Raw replies, queries and hashes are retained.
"""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures as cf
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'raw/novelty_audit_20261003'
DEST.mkdir(exist_ok=True)
TARGETS = [
    dict(sid=80364427, gid='3915674026806527616', oid=10168140,
         ra=170.45311084, dec=10.65945823),
    dict(sid=65258778, gid='1974721783974773376', oid=15865864,
         ra=327.53783409, dec=47.21569882)]
SIMBAD = 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync'
VIZIER = 'https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync'
GAIA = 'https://gea.esac.esa.int/tap-server/tap/sync'


def tap(label, url, query):
    return dict(label=label, url=url, format='csv',
                params=dict(REQUEST='doQuery', LANG='ADQL', FORMAT='csv',
                            QUERY=query))


def retrieve(task):
    row = dict(task)
    path = DEST / (task['label'] + '.' + task['format'])
    try:
        if not path.exists():
            response = requests.get(task['url'], params=task['params'], timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
        blob = path.read_bytes()
        row.update(file=str(path.relative_to(ROOT)), bytes=len(blob),
                   sha256=hashlib.sha256(blob).hexdigest())
        if task['format'] == 'csv':
            assert not blob.lstrip().startswith(b'<'), blob[:150]
            df = pd.read_csv(io.BytesIO(blob), dtype=str)
            records = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')
            row['columns'] = df.columns.tolist()
        else:
            root = ET.fromstring(blob)
            assert root.tag.split('}')[-1].upper() == 'VOTABLE', root.tag
            infos = [{**x.attrib, 'text': x.text} for x in root.iter()
                     if x.tag.split('}')[-1].upper() == 'INFO']
            assert not any(x.get('value', '').upper() == 'ERROR' for x in infos), infos
            fields = [x.attrib.get('ID', x.attrib.get('name')) for x in root.iter()
                      if x.tag.split('}')[-1].upper() == 'FIELD']
            records = []
            for x in root.iter():
                if x.tag.split('}')[-1].upper() != 'TR':
                    continue
                values = [y.text for y in x if y.tag.split('}')[-1].upper() == 'TD']
                assert len(fields) == len(values)
                records.append(dict(zip(fields, values)))
            row.update(columns=fields, infos=infos)
        row.update(status='valid_response', n_rows=len(records), records=records)
    except Exception as exc:
        row.update(status='failed_not_an_absence', error=repr(exc))
    return row


def main():
    tasks = []
    for t in TARGETS:
        sid, oid, ra, dec = t['sid'], t['oid'], t['ra'], t['dec']
        tasks += [tap(f'{sid}_simbad_aliases', SIMBAD,
                      f'SELECT id FROM ident WHERE oidref={oid}'),
                  tap(f'{sid}_simbad_references', SIMBAD,
                      'SELECT r.bibcode,r.title,r.abstract FROM has_ref h JOIN ref r '
                      f'ON h.oidbibref=r.oidbib WHERE h.oidref={oid}'),
                  tap(f'{sid}_vizier_vsx_60arcsec', VIZIER,
                      'SELECT * FROM "B/vsx/vsx" WHERE 1=CONTAINS('
                      f'POINT(\'ICRS\',RAJ2000,DEJ2000),CIRCLE(\'ICRS\',{ra},{dec},{60/3600}))')]
        tasks.append(dict(label=f'{sid}_aavso_vsx_60arcsec',
                          url='https://vsx.aavso.org/index.php', format='xml',
                          params=dict(view='query.votable', coords=f'{ra} +{dec}',
                                      format='d', size=60, unit=3, geom='r',
                                      order=9, filter='0,1,2')))
    # A named known variable is a positive control for the live VSX endpoint.
    tasks.append(dict(label='rr_lyr_aavso_vsx_positive_control',
                      url='https://vsx.aavso.org/index.php', format='xml',
                      params=dict(view='query.votable', ident='RR Lyr', filter='0,1,2')))
    ids = ','.join(t['gid'] for t in TARGETS)
    for table in ['vari_classifier_result', 'vari_short_timescale', 'vari_rotation_modulation']:
        tasks.append(tap('gaia_' + table, GAIA,
                         f'SELECT * FROM gaiadr3.{table} WHERE source_id IN ({ids})'))
    tasks.append(tap('gaia_variability_flags', GAIA,
                     'SELECT source_id,phot_variable_flag,has_epoch_photometry '
                     f'FROM gaiadr3.gaia_source WHERE source_id IN ({ids})'))
    rows = []
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(retrieve, tasks):
            rows.append(row)
            print(row['label'], row['status'], row.get('n_rows'),
                  row.get('error', ''), flush=True)
            output = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                          targets=TARGETS, results=rows,
                          limitation='A negative catalogue query is not universal proof of novelty.')
            (ROOT / 'data/novelty_audit_20261003.json').write_text(
                json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
