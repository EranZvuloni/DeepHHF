import numpy as np
import torch
from torchvision.transforms import Compose


class ApplyConstantLength:
    """
    Either pad or trim a signal to a constant desired length (hours)
    """

    def __init__(self, fs, hours=24):
        self.desired_length = fs * 60 * 60 * hours  # in samples

    def __call__(self, signal):
        current_length = signal.shape[0]
        if current_length < self.desired_length:  # Pad
            pad_length = self.desired_length - current_length
            padding_width = [(0, pad_length,)] + [(0, 0,)] * (signal.ndim - 1)
            return np.pad(signal, padding_width, mode='constant')
        else:  # Trim
            return signal[:self.desired_length]


class SegmentSample:
    """
    Sample a certain segment in each certain time from the recording
    """

    def __init__(self, fs, segment_length=5, sampling_rate=30, bsqi_threshold=0.8, fluctuate_startpoint=False, deterministic_startpoint=False):
        """

        :param fs: The signal sampling frequency. (Hz)
        :param segment_length: The length of a single sequence to extract. (minutes)
        :param sampling_rate: Divide the original signal by this rate and extract a segment from each part. (minutes)
        :param bsqi_threshold: Select the extracted signal above this threshold.
        :param fluctuate_startpoint: If True, random the starting point in each segment; otherwise, each segment starts
        in the same point within the interval and thus the interval remains constant.
        :param deterministic_startpoint: Only works with fluctuate_startpoint=False.
        This makes the startpoint in the segment become fixed, so it can be set by the user instead being randomized.
        This was added to allow a test time augmentation.
        The parameter needs to be a number between 0 and 1,
        determining the position along the segment - start to end, respectively.
        """
        # Convert all constants to sample units
        samples_per_min = fs * 60
        self.segment_length = int(segment_length * samples_per_min)
        self.sampling_rate = int(sampling_rate * samples_per_min)
        self.bsqi_threshold = bsqi_threshold
        self.fluctuate_startpoint = fluctuate_startpoint
        self.deterministic_startpoint = deterministic_startpoint

    def __call__(self, signal):

        nb_parts = int(signal.shape[0] / self.sampling_rate)

        all_bools = np.zeros(signal.shape[0], dtype='bool')

        # Randomly select the segment inside the parts lengths:
        if self.fluctuate_startpoint:
            starting_points = np.random.randint(0, self.sampling_rate-self.segment_length-1, nb_parts, dtype='int')
        else:
            if self.deterministic_startpoint:
                starting_points = np.floor(self.deterministic_startpoint*(self.sampling_rate-self.segment_length-1)).astype('int')
            else:
                starting_points = np.random.randint(0, self.sampling_rate-self.segment_length-1, 1, dtype='int')
            starting_points = np.repeat(starting_points, nb_parts)

        starting_points += np.cumsum(np.ones(nb_parts,
                                             dtype='int') * self.sampling_rate) - self.sampling_rate  # shift for the whole recording position
        for i in range(nb_parts):
            all_bools[starting_points[i]: starting_points[i] + self.segment_length] = 1

        segmented_signal = signal[all_bools, :]

        return segmented_signal


class BasicHolterTransform:
    """
    Basic transforms for the Holter signals:
    1 - from double to single.
    2 - from numpy array to torch tensor.
    3 - transpose: (L,C) --> (C,L)
    """

    def __init__(self, bfloat16=True):
        self.bfloat16 = bfloat16

    def __call__(self, signal):
        if self.bfloat16:
            t_signal = torch.tensor(signal, dtype=torch.bfloat16)  # bfloat16 makes the tensor less space consuming
        else:
            t_signal = torch.tensor(signal, dtype=torch.float32)
        t_signal = torch.transpose(t_signal, 0, 1)
        return t_signal


class Preprocessor:
    def __init__(self):
        transforms = []
        transforms += [ApplyConstantLength(128, hours=24)]
        transforms += [SegmentSample(128, segment_length=0.5, sampling_rate=2,
                                     fluctuate_startpoint=False,
                                     deterministic_startpoint=False)]
        transforms += [BasicHolterTransform(bfloat16=True)]
        self.transform = Compose(transforms)

    def __call__(self, signal):
        t_signal = self.transform(signal)
        t_signal = t_signal.unsqueeze(0)  # add batch dim
        return t_signal
