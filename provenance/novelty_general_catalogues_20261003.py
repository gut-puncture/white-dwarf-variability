"""Position-based prior-report checks in four published variable-star tables."""
import concurrent.futures as cf
import json
from novelty_audit_20261003 import ROOT, TARGETS, VIZIER, tap, retrieve

TABLES = [('atlas_variables', 'J/AJ/156/241/table4'),
          ('catalina_periodic', 'J/ApJS/213/9/table3'),
          ('ztf_variables', 'J/ApJS/249/18/table2'),
          ('ztf_suspected', 'J/ApJS/249/18/table3')]


def main():
    tasks = [tap('schema_' + label, VIZIER, f'SELECT TOP 1 * FROM "{table}"')
             for label, table in TABLES]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        schemas = list(pool.map(retrieve, tasks))
    queries = []
    for (label, table), schema in zip(TABLES, schemas):
        assert schema['status'] == 'valid_response', schema
        columns = schema['columns']
        coordinate_pairs = [('RAJ2000', 'DEJ2000'), ('RA_ICRS', 'DE_ICRS'),
                            ('RAdeg', 'DEdeg'), ('RA', 'Dec'), ('ra', 'dec')]
        available = [(ra, dec) for ra, dec in coordinate_pairs
                     if ra in columns and dec in columns]
        assert len(available) == 1, (label, columns, available)
        ra, dec = available[0]
        print('COLUMNS', label, ra, dec, flush=True)
        for t in TARGETS:
            query = (f'SELECT * FROM "{table}" WHERE 1=CONTAINS('
                     f'POINT(\'ICRS\',"{ra}","{dec}"),'
                     f'CIRCLE(\'ICRS\',{t["ra"]},{t["dec"]},{60/3600}))')
            queries.append(tap(f'{t["sid"]}_{label}_60arcsec', VIZIER, query))
    rows = []
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(retrieve, queries):
            rows.append(row)
            print(row['label'], row['status'], row.get('n_rows'),
                  row.get('error', ''), flush=True)
    (ROOT / 'data/novelty_general_catalogues_20261003.json').write_text(
        json.dumps(dict(schemas=schemas, results=rows), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
