import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossCompressUnit():
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.weight_vv = nn.Parameter(torch.FloatTensor(dim, 1))
        self.weight_ev = nn.Parameter(torch.FloatTensor(dim, 1))
        self.weight_ve = nn.Parameter(torch.FloatTensor(dim, 1))
        self.weight_ee = nn.Parameter(torch.FloatTensor(dim, 1))
        self.bias_v = nn.Parameter(torch.FloatTensor(dim))
        self.bias_e = nn.Parameter(torch.FloatTensor(dim))

    def __call__(self, v, e):
        v = torch.unsqueeze(v, dim=2)
        e = torch.unsqueeze(e, dim=1)

        c_matrix = torch.matmul(v ,e)
        c_matrix_transpose = c_matrix.transpose(1,2)

        c_matrix = c_matrix.reshape((-1, self.dim))
        c_matrix_transpose = c_matrix_transpose.reshape((-1, self.dim))

        v_output = torch.reshape(torch.matmul(c_matrix, self.weight_vv) + torch.matmul(c_matrix_transpose, self.weight_ev), (-1, self.dim)) + self.bias_v
        e_output = torch.reshape(torch.matmul(c_matrix, self.weight_ve) + torch.matmul(c_matrix_transpose, self.weight_ee), (-1, self.dim)) + self.bias_e
        return v_output, e_output

