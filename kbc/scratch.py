'''
This is script for running tests
'''


from pathlib import Path
import pkg_resources
import pickle
import numpy as np
import torch
from kbc.models import KBCModel
from typing import Dict, Tuple, List


DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))
name = 'WN18RR'
root = DATA_PATH / name

print(root)

f = 'train' #test, valid
data = {}

for f in ['train','test','valid']:
    in_file = open(str(root / (f + '.pickle')), 'rb')
    data[f] = pickle.load(in_file)


maxis = np.max(data['train'], axis=0)
n_entities = int(max(maxis[0], maxis[2]) + 1)
n_predicates = int(maxis[1] + 1)
n_predicates *= 2

# inp_f = open(str(root / f'to_skip.pickle'), 'rb')
# to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
# inp_f.close()

inp_f = open(str(root / f'to_skip.pickle'), 'rb')
x = pickle.load(inp_f)
inp_f.close()

copy = np.copy(data['train'])
tmp = np.copy(copy[:, 0])
copy[:, 0] = copy[:, 2]
copy[:, 2] = tmp
copy[:, 1] += n_predicates // 2  # has been multiplied by two.
ans = np.vstack((data['train'], copy))


def get_examples(data, split):
    return data[split]


def eval(
        model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
        at: Tuple[int] = (1, 3, 10)):

    test = get_examples(data, split)
    examples = torch.from_numpy(test.astype('int64')).cpu()  # .cuda()
    missing = [missing_eval]
    if missing_eval == 'both':
        missing = ['rhs', 'lhs']

    mean_reciprocal_rank = {}
    hits_at = {}

    for m in missing:
        q = examples.clone()
        if n_queries > 0:
            permutation = torch.randperm(len(examples))[:n_queries]
            q = examples[permutation]
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += n_predicates // 2
        ranks = model.get_ranking(q, to_skip[m], batch_size=500)
        mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
        hits_at[m] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            at
        )))).cpu()

    return mean_reciprocal_rank, hits_at

n_queries = 10

test = data['train']
examples = torch.from_numpy(test.astype('int64')).cpu()
q = examples.clone()

permutation = torch.randperm(len(examples))[:n_queries]
