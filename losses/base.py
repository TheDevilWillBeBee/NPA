import torch

class BaseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.info = {}

    def set_info(self, value, **terms):
        self.info.clear()
        self.info.update(value=value)
        self.info.update(terms=terms)

    def set_summary(self, **summary):
        self.info.update(summary=summary)
    
    def get_info(self):
        return self.info
    
    def clear_info(self):
        self.info.clear()

    def forward(self, *args):
        raise NotImplementedError()
