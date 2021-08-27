class Messenger(object):
    def __init__(self, bootstrap_server: str):
        self._bootstrap_server = bootstrap_server

    def send(self, data, tag: str, suffix=None):
        pass

    def receive(self, tag: str, suffix=None):
        pass

    @property
    def bootstrap_server(self):
        return self._bootstrap_server

    @staticmethod
    def generate_message_tag(tag, suffix):
        if suffix is None:
            return tag
        suffix_str = [str(element) for element in suffix]
        return tag + '.' + '.'.join(suffix_str)
