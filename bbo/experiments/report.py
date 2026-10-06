"""Score a trials.jsonl file using explicit, matched loss-space references."""
import argparse
import json
from pathlib import Path
from .scoring import score_losses, VERSION


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trials', type=Path)
    p.add_argument('--family', required=True, choices=('BBOB','HPO','DBTune','BBOPlace','GuacaMol'))
    p.add_argument('--metric', required=True)
    p.add_argument('--direction', required=True, choices=('minimize','maximize'))
    p.add_argument('--initial', required=True, type=int)
    p.add_argument('--budget', required=True, type=int)
    p.add_argument('--gp-loss', type=float, help='Matched GP final loss, in minimization orientation')
    p.add_argument('--upper-loss', type=float, help='Theoretical upper-quality reference, in loss orientation')
    p.add_argument('--carry-forward', action='store_true', help='Explicitly fill an early-ended trajectory with its incumbent')
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in args.trials.read_text().splitlines() if line.strip()]
    if [r['trial_id'] for r in rows] != list(range(len(rows))) or any(r['status'] != 'success' for r in rows):
        raise ValueError('Scoring requires ordered successful records; do not silently drop failed evaluations')
    losses = [(1 if args.direction == 'minimize' else -1) * float(r['objectives'][args.metric]) for r in rows]
    result, checkpoints, reference = score_losses(losses, args.initial, args.budget,
        args.upper_loss, args.gp_loss, family=args.family, early_end=args.carry_forward)
    print(json.dumps(dict(version=VERSION, result=result, normalized_checkpoints=checkpoints, reference=reference), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
