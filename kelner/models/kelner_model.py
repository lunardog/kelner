
class KelnerModel(object):

    def __init__(self):
        self.input = None
        self.output = None
        pass

    def __call__(self, data):
        """
        Runs inference on a batch of data
        """
        raise NotImplemented()

    def summary(self):
        pass
