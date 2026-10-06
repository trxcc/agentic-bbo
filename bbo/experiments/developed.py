"""Load a selected optimizer without changing its source or calling an LLM."""
import hashlib
import importlib.util
import sys
from .tasks import ASSETS
from .io import read


def load_selected_optimizer(name):
    root = ASSETS / 'developed_optimizers'
    rows = [r for r in read(root/'inventory.json') if r['program_file'].removesuffix('.py') == name]
    if len(rows) != 1:
        raise ValueError('Unknown frozen optimizer program')
    path = root / rows[0]['program_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != rows[0]['sha256']:
        raise ValueError('Selected optimizer source changed')
    support_spec = importlib.util.spec_from_file_location('solver_support', root/'solver_support.py')
    support = importlib.util.module_from_spec(support_spec)
    support_spec.loader.exec_module(support)
    previous = sys.modules.get('solver_support')
    sys.modules['solver_support'] = support
    try:
        spec = importlib.util.spec_from_file_location('selected_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.create_optimizer()
    finally:
        if previous is None:
            sys.modules.pop('solver_support', None)
        else:
            sys.modules['solver_support'] = previous
