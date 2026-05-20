from torch.utils.data import Dataset
from .base_data_set import BaseDataSet, enumerate_development_datasets
import copy
import numpy as np
import torch

VALID_TYPES = {
    'same_mic_same_type': {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
    },
    'same_mic_all_types': {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3],
        3: [0, 1, 2, 3],
    },
}


class ComplementMCMDataSet(Dataset):
    def __init__(
            self,
            num_samples=None,
            valid_types='same_mic_all_types', # 'same_mic_same_type' or 'same_mic_all_types'
            **kwargs
    ):
        super(ComplementMCMDataSet, self).__init__()

        assert type(kwargs['machine_type']) == int and type(kwargs['machine_id']) == int
        assert kwargs['machine_id'] >= 0
        assert kwargs['machine_type'] >= 0

        self.kwargs = kwargs

        training_sets = []

        for type_ in VALID_TYPES[valid_types][kwargs['machine_type']]:
            for id_ in [0, 2, 4, 6]:
                if type_ != kwargs['machine_type'] or id_ != kwargs['machine_id']:
                    kwargs_ = copy.deepcopy(kwargs)
                    kwargs_['machine_type'] = type_
                    kwargs_['machine_id'] = id_
                    t = BaseDataSet(label=1, **kwargs_)
                    training_sets.append(t)

        self.training_set = torch.utils.data.ConcatDataset(training_sets)

        if num_samples is not None:
            assert num_samples > 0
            num_samples = min(num_samples, len(self.training_set))

            indices = np.random.choice(len(self.training_set), size=num_samples, replace=False)

            if num_samples < 64:
                indices = np.random.choice(indices, size=64, replace=True)

            self.training_set = torch.utils.data.Subset(self.training_set, indices)

if __name__ == '__main__':
    for snr_, type_, id_, normal_ in enumerate_development_datasets():
        dataset = ComplementMCMDataSet(num_samples=1024, data_path="/Users/zhang/MuReNN/data/MIMII",
                                 snr=snr_, machine_type=type_, machine_id=id_, normal=normal_)
        print(len(dataset.training_data_set()))