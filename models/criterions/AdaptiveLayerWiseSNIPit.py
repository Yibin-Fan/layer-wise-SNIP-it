from models.criterions.AdaptiveLayerWiseSNIP import AdaptiveLayerWiseSNIP


class AdaptiveLayerWiseSNIPit(AdaptiveLayerWiseSNIP):

    """
    Iterative adaptive layer-wise SNIP before training.
    """

    def __init__(self, *args, limit=0.0, steps=5, **kwargs):
        self.limit = limit
        super(AdaptiveLayerWiseSNIPit, self).__init__(*args, **kwargs)
        self.steps = [limit - (limit - 0.5) * (0.5 ** i) for i in range(steps + 1)] + [limit]

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage=0.0, *args, **kwargs):
        while len(self.steps) > 0:
            percentage = self.steps.pop(0)
            super().prune(percentage=percentage, *args, **kwargs)
