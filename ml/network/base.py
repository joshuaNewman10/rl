MODEL_REGISTRY = {}


class NetworkNotFoundError(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload  # you could add more args

    def __str__(self):
        return str(self.message)


class NetworkFactory:
    def get_network(self, network_name: str, network_params=None):
        try:
            model_cls = MODEL_REGISTRY[network_name]
        except KeyError:
            raise NetworkNotFoundError(f"Unsupported network {network_name}")

        return model_cls(network_params)
