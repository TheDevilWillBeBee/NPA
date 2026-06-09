import torch

state_dict = torch.load("data/pretrained/polka_dotted.pth")
print(state_dict.keys())

#  Remove the N0 key from the state dict
if "N0" in state_dict:
    del state_dict["N0"]

torch.save(state_dict, "data/pretrained/polka_dotted.pth")