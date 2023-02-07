import torch
import torchvision.models as models

from typing import Any
from .custom import Custom
import sys
sys.path.append('/home/yifan/dataset/clean_ref/pairflip/cifar10/0')
import Model.model as subject_model



class Resnet18_ecoset(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.model_path = parameters.get('model_path', '/home/yifan/dataset/clean_ref/pairflip/cifar10/0/Model/Epoch_199/subject_model.pth')

    def create_model(self) -> Any:
        # model = models.resnet50(weights=None, num_classes=10)
        model = eval("subject_model.{}()".format('resnet18'))
        # path_to_weights = "https://osf.io/gd9kn/download"
        # state_dict = torch.hub.load_state_dict_from_url(
        #     path_to_weights, map_location=self.device, file_name="Resnet50_ecoset"
        # )
        state_dict= torch.load(self.model_path)
        model.load_state_dict(state_dict)
        return model, None
