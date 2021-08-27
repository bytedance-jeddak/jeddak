class FedAveParameter(object):
    """
    FedAve algorithm parameters base class
    """
    def __init__(self, client_frac, local_epoch_num, local_batch_size):
        """

        :param client_frac: the fraction of clients selected to update the global model
        :param local_epoch_num: the number of local epochs
        :param local_batch_size: the number of local batch size
        """
        self.client_frac = client_frac
        self.local_epoch_num = local_epoch_num
        self.local_batch_size = local_batch_size
