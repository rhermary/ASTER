import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/model")

from ts_benchmark.baselines.LLM.model import ASTER

class ASTERModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.model = ASTER.ASTER(window_size=config.seq_len, num_features=config.llm_dim, signal_num=config.c_out)

    def forward(self, x_enc):        
        output = self.model.training_step(x_enc)
        return output
