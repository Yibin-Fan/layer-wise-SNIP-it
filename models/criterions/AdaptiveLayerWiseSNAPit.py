from models.criterions.AdaptiveLayerWiseSNAP import AdaptiveLayerWiseSNAP


class AdaptiveLayerWiseSNAPit(AdaptiveLayerWiseSNAP):

    """
    Iterative adaptive layer-wise SNAP before training.
    """

    def __init__(self, *args, limit=0.0, start=0.5, steps=5, **kwargs):
        self.limit = limit
        super(AdaptiveLayerWiseSNAPit, self).__init__(*args, **kwargs)
        self.steps = [limit - (limit - start) * (0.5 ** i) for i in range(steps + 1)] + [limit]
        self.left = 1.0
        self.pruned = 0.0

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        while len(self.steps) > 0:
            percentage = self.steps.pop(0)
            prune_now = (percentage - self.pruned) / (self.left + 1e-8)
            super().prune(percentage=prune_now, *args, **kwargs)
            self.pruned = self.model.structural_sparsity
            self.left = 1.0 - self.pruned
