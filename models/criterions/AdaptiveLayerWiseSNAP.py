import torch

from models.criterions.LayerWiseSNAP import LayerWiseSNAP
from utils.data_utils import lookahead_finished


class AdaptiveLayerWiseSNAP(LayerWiseSNAP):

    """
    Structured layer-wise SNAP with adaptive per-layer pruning allocation.
    Layers still rank nodes internally, but the amount pruned per layer is
    assigned from layer-level sensitivity with a minimum keep constraint.
    """

    min_keep_ratio = 0.15
    min_keep_nodes = 8

    def _get_layer_keep_vectors(self, grads_abs, norm_factor, percentage):
        entries = self._collect_layer_entries(grads_abs, norm_factor)
        prune_counts = self._allocate_prune_counts(entries, percentage)
        return {
            entry["key"]: self._build_keep_vector(entry, prune_counts[entry["key"]])
            for entry in entries
        }

    def _collect_layer_entries(self, grads_abs, norm_factor):
        entries = []
        seen = set()
        for ((identification, name), grad), (first, last) in lookahead_finished(grads_abs.items()):
            if identification in seen:
                continue
            seen.add(identification)
            scores = (grad / norm_factor).detach().flatten()
            total = scores.numel()
            if (first or last) and not self.model._outer_layer_pruning:
                max_prune = 0
            else:
                min_keep = min(total, max(self.min_keep_nodes, int(total * self.min_keep_ratio)))
                max_prune = max(total - min_keep, 0)
            entries.append({
                "key": identification,
                "scores": scores,
                "total": total,
                "max_prune": max_prune,
                "importance": scores.mean().item() if total > 0 else 0.0,
            })
        return entries

    def _allocate_prune_counts(self, entries, percentage):
        total_nodes = sum(entry["total"] for entry in entries)
        target_prune = int(total_nodes * percentage)
        total_capacity = sum(entry["max_prune"] for entry in entries)
        target_prune = min(max(target_prune, 0), total_capacity)

        prune_counts = {entry["key"]: 0 for entry in entries}
        if target_prune == 0:
            return prune_counts

        capacities = torch.tensor([entry["max_prune"] for entry in entries], dtype=torch.float32, device=self.device)
        importances = torch.tensor([entry["importance"] for entry in entries], dtype=torch.float32, device=self.device)
        weights = capacities / importances.clamp_min(1e-12)
        if weights.sum().item() <= 0:
            weights = capacities

        raw = target_prune * (weights / (weights.sum() + 1e-12))
        allocated = torch.floor(raw).long()
        allocated = torch.minimum(allocated, capacities.long())

        leftover = int(target_prune - allocated.sum().item())
        if leftover > 0:
            fractional = raw - allocated.float()
            order = torch.argsort(fractional, descending=True)
            while leftover > 0:
                progressed = False
                for idx in order.tolist():
                    if leftover == 0:
                        break
                    if allocated[idx].item() >= capacities[idx].item():
                        continue
                    allocated[idx] += 1
                    leftover -= 1
                    progressed = True
                if not progressed:
                    break

        for entry, count in zip(entries, allocated.tolist()):
            prune_counts[entry["key"]] = count
        return prune_counts

    def _build_keep_vector(self, entry, prune_count):
        keep_count = entry["total"] - prune_count
        keep_count = min(max(keep_count, 1), entry["total"])
        keep_vector = torch.zeros_like(entry["scores"], dtype=torch.float32, device=self.device)
        _, keep_indices = torch.topk(entry["scores"], keep_count, sorted=False)
        keep_vector[keep_indices] = 1.0
        return keep_vector
