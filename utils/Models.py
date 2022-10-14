import torch.nn as nn 
import timm 
import torchvision.models as models 




class Model(nn.Module):
    def __init__(self,model_name,num_classes):
        super(Model,self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.fc = nn.Linear(in_features=1000,out_features=num_classes)
    def forward(self,x):
        x = self.model(x)
        x = self.fc(x)
        return x 

class Cait(nn.Module):
    def __init__(self,num_classes):
        super(Cait,self).__init__()
        self.model = timm.create_model('cait_s36_384', pretrained=True, num_classes=num_classes)
    def forward(self,x):
        x = self.model(x)
        return x 

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x