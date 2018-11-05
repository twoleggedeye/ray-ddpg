import ray


@ray.remote
class StorageWrapper(object):
    def __init__(self, storage_prototype, storage_args, storage_kwargs):
        self.storage = storage_prototype(*storage_args, **storage_kwargs)

    def add(self, *args):
        self.storage.add(*[arg.copy() for arg in args])
        del args
    
    def sample(self, batch_size):
        return self.storage.sample(min(batch_size, len(self.storage)))

    def get_len(self):
        return len(self.storage)


@ray.remote
class ParamServer(object):
    def __init__(self, initial_handle):
        self._state_dict = None
        self.push(initial_handle)

    def push(self, weight_handle):
        self._state_dict = weight_handle

    def pull(self):
        return self._state_dict
