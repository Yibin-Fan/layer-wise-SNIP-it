import numpy as np
import torch
from torch import nn

from models.criterions.SNAP import SNAP
from utils.data_utils import lookahead_finished


class LayerWiseSNAP(SNAP):

    """
    Structured SNAP variant that ranks governing-node sensitivities inside
    each layer instead of using one global threshold across all layers.
    """

    def handle_pruning(self, all_scores, grads_abs, norm_factor, percentage):
        summed_weights = sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "weight" in name])

        summed_pruned = 0
        toggle_row_column = True
        cutoff = 0
        length_nonzero = 0
        for ((identification, name), grad), (first, last) in lookahead_finished(grads_abs.items()):
            corresponding_weight_parameter = [val for key, val in self.model.named_parameters() if key == name][0]
            is_conv = len(corresponding_weight_parameter.shape) > 2
            corresponding_module: nn.Module = \
                [val for key, val in self.model.named_modules() if key == name.split(".weight")[0]][0]

            binary_keep_neuron_vector = self._get_layer_keep_vector(
                grad=grad,
                norm_factor=norm_factor,
                percentage=percentage,
                first=first,
                last=last,
            )

            if first or last:
                length_nonzero = self.handle_outer_layers(binary_keep_neuron_vector,
                                                          first,
                                                          is_conv,
                                                          last,
                                                          length_nonzero,
                                                          corresponding_module,
                                                          name,
                                                          corresponding_weight_parameter)
            else:
                cutoff, length_nonzero = self.handle_middle_layers(binary_keep_neuron_vector,
                                                                   cutoff,
                                                                   is_conv,
                                                                   length_nonzero,
                                                                   corresponding_module,
                                                                   name,
                                                                   toggle_row_column,
                                                                   corresponding_weight_parameter)

            cutoff, summed_pruned = self.print_layer_progress(cutoff,
                                                              grads_abs,
                                                              length_nonzero,
                                                              name,
                                                              summed_pruned,
                                                              toggle_row_column,
                                                              corresponding_weight_parameter)
            toggle_row_column = not toggle_row_column

        for line in str(self.model).split("\n"):
            if "BatchNorm" in line or "Conv" in line or "Linear" in line or "AdaptiveAvg" in line or "Sequential" in line:
                print(line)
        print("final percentage after layer-wise snap:", summed_pruned / summed_weights)

        self.model.apply_weight_mask()
        self.cut_lonely_connections()

    def _get_layer_keep_vector(self, grad, norm_factor, percentage, first, last):
        if (first or last) and not self.model._outer_layer_pruning:
            return torch.ones_like(grad).float().to(self.device)

        scores = (grad / norm_factor).detach().flatten()
        keep_count = int(scores.numel() * (1 - percentage))
        if keep_count < 1:
            keep_count = 1
        elif keep_count > scores.numel():
            keep_count = scores.numel()

        keep_vector = torch.zeros_like(scores, dtype=torch.float32, device=self.device)
        _, keep_indices = torch.topk(scores, keep_count, sorted=False)
        keep_vector[keep_indices] = 1.0
        return keep_vector.view_as(grad)
