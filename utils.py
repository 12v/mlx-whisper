from contextlib import contextmanager

import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class DummyWandb:
    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


@contextmanager
def conditional_autocast():
    if torch.amp.autocast_mode.is_autocast_available(device.type):
        with torch.autocast(device.type):
            yield
    else:
        yield
