from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet as OldResNet
from torchvision.models.resnet import ResNet50_Weights, Bottleneck

from models.self_attention import SelfAttention


class ResNetAttn(OldResNet):
    def __init__(self,
                 config,
                 block = Bottleneck,
                 layers = [3, 4, 6, 3],
                 weights = ResNet50_Weights.IMAGENET1K_V1,
                 **kwargs):
        super().__init__(block, layers)
        self.load_state_dict(weights.get_state_dict(check_hash=True))

        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        del self.fc
        del self.avgpool

        self.self_attn = SelfAttention(2048, 3, 0.2, "tanh")

        self.bn = torch.nn.BatchNorm1d(2048)
        torch.nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):
        # x: (b, c, t, h, w)
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        # 2D ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_max_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # (b*t, 2048)
        x = rearrange(x, '(b t) c -> b t c', b=b, t=t)  # (b, t, 2048)

        # Self-attention
        x, x_score = self.self_attn(x)
        x = self.bn(x)

        return x
