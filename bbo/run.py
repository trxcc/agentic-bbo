"""Default CLI: the paper's Docker agent, with the full benchmark inventory."""
from .experiments.run import main as _main


def main(argv=None):
    return _main(argv, default_suite='main')


if __name__ == '__main__':
    raise SystemExit(main())
