from models.criterions.LayerWiseSNIP import LayerWiseSNIP
from models.criterions.LayerWiseSNIPit import LayerWiseSNIPit


class LayerWiseSNIPitDuring(LayerWiseSNIPit):

    """
    Iterative layer-wise SNIP during training.
    """

    def __init__(self, *args, **kwargs):
        super(LayerWiseSNIPitDuring, self).__init__(*args, **kwargs)

    def prune(self, percentage=0.0, *args, **kwargs):
        if len(self.steps) > 0:
            percentage = self.steps.pop(0)
            kwargs["percentage"] = percentage
            LayerWiseSNIP.prune(self, **kwargs)
