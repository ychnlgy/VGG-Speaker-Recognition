import math

import keras
import numpy


class Dataset:
    """Slices spectrogram across steps.

    Note that in this implementation, the end might be sliced off.
    Choose a small step size to reduce the wasted amount.
    """

    def __init__(self, spec, slice_size, step_size):
        spec = spec.squeeze()
        assert len(spec.shape) == 2, spec.shape
        self._spec = numpy.expand_dims(spec, 0)  # (1, freq, time)
        self._slice_size = slice_size
        self._step_size = step_size
        self._dt = spec.shape[1]
        self._len = (self._dt - self._slice_size) // self._step_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        i = idx * self._step_size
        j = i + self._slice_size
        return self._spec[:, :, i:j]  # (1, freq, time)

class DataLoader(keras.utils.Sequence):

    def __init__(self, dataset, batch_size):
        self.dset = dataset
        self.batch = batch_size
        self._len = math.ceil(len(self.dset) / self.batch)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        out = []
        for di in range(self.batch):
            idx = i + di
            if idx < len(self.dset):
                out.append(self.dset[idx])
            else:
                break
        return numpy.concatenate(out, axis=0)
