import torch

from models.criterions.LayerWiseSNIP import LayerWiseSNIP


class AdaptiveLayerWiseSNIP(LayerWiseSNIP):

    """
    Layer-wise SNIP variant that allocates the global keep budget using
    layer-level average sensitivity, then ranks weights within each layer.
    """

    def _get_keep_counts(self, entries, percentage):
        if len(entries) == 0:
            return {}

        total_params = sum(entry["total"] for entry in entries)
        target_active = self._clamp_keep_count(int(total_params * (1 - percentage)), total_params)
        current_active = sum(entry["active_indices"].numel() for entry in entries)
        prune_now = max(current_active - target_active, 0)

        keep_counts = {entry["name"]: entry["active_indices"].numel() for entry in entries}
        if prune_now <= 0:
            return keep_counts

        prunable = torch.tensor(
            [max(entry["active_indices"].numel() - 1, 0) for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        prune_now = min(prune_now, int(prunable.sum().item()))
        if prune_now <= 0:
            return keep_counts

        layer_stats = torch.tensor(
            [self._layer_sensitivity(entry) for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        weights = prunable / layer_stats.clamp_min(1e-12)
        if weights.sum().item() <= 0:
            weights = prunable

        raw_prune = prune_now * (weights / (weights.sum() + 1e-12))
        prune_counts = torch.floor(raw_prune).long()
        prune_counts = torch.minimum(prune_counts, prunable.long())

        leftover = int(prune_now - prune_counts.sum().item())
        if leftover > 0:
            fractional = raw_prune - prune_counts.float()
            order = torch.argsort(fractional, descending=True)
            for idx in order.tolist():
                if leftover == 0:
                    break
                if prune_counts[idx].item() >= prunable[idx].item():
                    continue
                prune_counts[idx] += 1
                leftover -= 1

        for entry, prune_count in zip(entries, prune_counts.tolist()):
            keep_counts[entry["name"]] -= prune_count
            keep_counts[entry["name"]] = self._clamp_keep_count(keep_counts[entry["name"]], entry["total"])

        return keep_counts

    @staticmethod
    def _layer_sensitivity(entry):
        if entry["active_scores"].numel() == 0:
            return 0.0
        return entry["active_scores"].mean().item()
