"""Verify frozen reference checksums without calling models or objectives."""
import hashlib
from pathlib import Path
from .tasks import ASSETS, PaperTask, task_rows
from .io import read


def main():
    root = Path(__file__).resolve().parents[2]
    sources = read(ASSETS / 'provenance.json')
    bundled = 0
    for relative, item in sources.items():
        if item.get('distributed') is False:
            continue
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Frozen reference changed: ' + relative)
        bundled += 1
    count = 0
    for suite in ('main', 'frontier', 'controlled'):
        for row in task_rows(suite):
            if suite == 'controlled':
                from .controlled import ControlledPriorTask
                task = ControlledPriorTask(row['task'], seed=row['seed'], prior=row['prior'])
            else:
                task = PaperTask(row['task'], suite=suite, seed=row['seed'])
            if not task.sanity_check().ok:
                raise ValueError('Task sanity failed: ' + row['task'])
            count += 1
    print(f'Verified {bundled} bundled frozen files and {count} task/seed contracts; no objective calls.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
