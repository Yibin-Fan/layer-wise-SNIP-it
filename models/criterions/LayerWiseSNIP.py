import os

import torch

from models.criterions.SNIP import SNIP
from utils.constants import RESULTS_DIR, OUTPUT_DIR


class LayerWiseSNIP(SNIP):

    """
    SNIP variant that keeps the original sensitivity statistic but applies
    pruning thresholds independently inside each layer.
    """

    def handle_pruning(self, all_scores, grads_abs, log10, manager, norm_factor, percentage):
        self._save_scores(all_scores, manager)

        entries = self._collect_layer_entries(grads_abs, norm_factor)
        keep_counts = self._get_keep_counts(entries, percentage)

        for entry in entries:
            old_active = entry["mask"].sum().item()
            new_mask = self._mask_from_keep_count(entry, keep_counts[entry["name"]])
            self.model.mask[entry["name"]] = new_mask.view_as(self.model.mask[entry["name"]]).to(self.device)

            length = float(entry["mask"].numel())
            cutoff = (self.model.mask[entry["name"]] == 0).sum().item()
            print(old_active)
            print("layer-wise pruning", entry["name"], cutoff, "percentage", cutoff / length, "length_nonzero", length)

        self.model.apply_weight_mask()
        print("final percentage after layer-wise snip:", self.model.pruned_percentage)
        self.cut_lonely_connections()

    def _save_scores(self, all_scores, manager):
        if manager is None:
            return
        manager.save_python_obj(all_scores.detach().cpu().numpy(),
                                os.path.join(RESULTS_DIR, manager.stamp, OUTPUT_DIR, "layer_wise_scores"))

    def _collect_layer_entries(self, grads_abs, norm_factor):
        entries = []
        for name, grad in grads_abs.items():
            scores = (grad / norm_factor).detach().flatten()
            mask = self.model.mask[name].detach().flatten().bool()
            active_indices = torch.nonzero(mask, as_tuple=False).flatten()
            active_scores = scores[active_indices]
            entries.append({
                "name": name,
                "scores": scores,
                "mask": mask,
                "active_indices": active_indices,
                "active_scores": active_scores,
                "total": scores.numel(),
            })
        return entries

    def _get_keep_counts(self, entries, percentage):
        keep_counts = {}
        for entry in entries:
            keep_count = int(entry["total"] * (1 - percentage))
            keep_counts[entry["name"]] = self._clamp_keep_count(keep_count, entry["total"])
        return keep_counts

    @staticmethod
    def _clamp_keep_count(keep_count, total):
        if total <= 0:
            return 0
        if keep_count < 1:
            return 1
        if keep_count > total:
            return total
        return keep_count

    def _mask_from_keep_count(self, entry, keep_count):
        keep_count = min(keep_count, entry["active_indices"].numel())
        new_mask = torch.zeros_like(entry["mask"], dtype=torch.float32, device=self.device)
        if keep_count <= 0:
            return new_mask

        _, local_keep = torch.topk(entry["active_scores"], keep_count, sorted=False)
        keep_indices = entry["active_indices"][local_keep]
        new_mask[keep_indices] = 1.0
        return new_mask
