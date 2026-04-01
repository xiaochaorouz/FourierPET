"""
Projection simulation utilities for parallel-beam PET.
"""
import numpy as np
from torch_radon import ParallelBeam
import torch
from tqdm.auto import tqdm


class PBean:
    def __init__(self, angles, det_count, volume):
        angles = np.linspace(
            angles[0] * np.pi, angles[1] * np.pi,
            angles[2], endpoint=False)
        self.pbeam = ParallelBeam(det_count, angles, volume=tuple(volume))

    def to_sino(self, ins):
        return self.pbeam.forward(torch.FloatTensor(ins).cuda()).cpu()

    def to_case_sino(self, case, progress_bar=False):
        case_sino = []
        for slice in tqdm(case, desc="\t Forward Projection",
                          unit="Slice"):
            case_sino.append(self.to_sino(slice).numpy())
        return np.stack(case_sino, axis=0)
