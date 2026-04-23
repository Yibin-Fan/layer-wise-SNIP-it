from models.criterions.AdaptiveLayerWiseSNIP import AdaptiveLayerWiseSNIP
from models.criterions.AdaptiveLayerWiseSNIPit import AdaptiveLayerWiseSNIPit


class AdaptiveLayerWiseSNIPitDuring(AdaptiveLayerWiseSNIPit):

    """
    Iterative adaptive layer-wise SNIP during training.
    """

    def __init__(self, *args, **kwargs):
        super(AdaptiveLayerWiseSNIPitDuring, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) > 0:
            percentage = self.steps.pop(0)
            kwargs["percentage"] = percentage
            AdaptiveLayerWiseSNIP.prune(self, **kwargs)
