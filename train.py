import torch
import torch.nn as nn
import torch.optim as optim

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from torchvision import transforms

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(True)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
    
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

class config:
    mm_vision_tower='openai/clip-vit-large-patch14'
    mm_vision_select_layer=-2
    mm_vision_select_feature='patch'
    
from transformers import CLIPProcessor, CLIPModel

clip = build_vision_tower(config)
clipProcessor = CLIPProcessor.from_pretrained(config.mm_vision_tower)

from PIL import Image
import requests


def display(tensor,path):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    stddev = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    normalized_image = tensor
    restored_image = normalized_image * stddev.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    to_pil = transforms.ToPILImage()
    tensor = (restored_image * 255).byte()
    pil_image = to_pil(tensor[0])
    pil_image.save(path)



# inputs shape torch.Size([1, 3, 224, 224])
# embedding shape torch.Size([1, 256, 1024])

# load a harmful images
image = Image.open('/home/changsheng/adv_vlm/bomb_224.png')
inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
x_harm = torch.tensor(inputs['pixel_values'], dtype=torch.float32)



# load a adv images (have their own meanings)

image = Image.open("/home/changsheng/adv_vlm/cat_224.png")
inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
x_adv = torch.tensor((inputs['pixel_values']).clone().detach(), dtype=torch.float32)

image = Image.open("/home/changsheng/adv_vlm/cat_224.png")
inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
x_ori = torch.tensor(inputs['pixel_values'].clone().detach(), dtype=torch.float32)

# the patch you can select: stationary and moving

print('------------------------- Start Running ----------')

# we first choose the stationary one 
patch_type = 'stationary'
patch_size = 40
start_x_y = (30,30)
mask_size = x_adv.shape[2]

    # generate a mask
if patch_type == 'stationary':
    print('Use stationary patch')
    x_position = start_x_y[0]
    y_position = start_x_y[1]
    mask = torch.zeros_like(x_adv)
    mask[:, :, x_position:x_position + patch_size, y_position:y_position + patch_size] = 1

elif patch_type == 'moving':
    
    print('Use moving patch')
    
else:
    print('TBD')

type_attack = 'adv'
if type_attack == 'adv':
    # x_adv = torch.randn(1, 3, 224, 224, requires_grad=True)
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    clip.to(device)
    x_adv.to(device)
    x_harm.to(device)
    
    # Load a optimize(Adam,..)
    learning_rate = 0.1
    
    stop_para = 0.1
    epoch_counter = 0
    loss_now = 9999
    
    optimizer = optim.Adam([x_adv.requires_grad_()], lr=learning_rate)  
    criterion = nn.MSELoss() # L2 Loss

    while loss_now > stop_para:
        optimizer.zero_grad()  
        h_adv = clip(x_adv)
        h_harm = clip(x_harm)
        loss = criterion(h_adv, h_harm)
        loss.backward()
        optimizer.step()
        loss_now = loss.item()
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        stddev = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        normalized_image = x_adv.data
        restored_image = normalized_image * stddev.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        restored_image = restored_image.clamp(0, 1)
        x_adv.data = (restored_image - mean.view(1, 3, 1, 1)) / stddev.view(1, 3, 1, 1)
        
        print(f'Epoch [{epoch_counter}], Loss: {loss.item()}')
        epoch_counter += 1
        
        if epoch_counter % 500 == 0:
            path = "adv_random_0.01.png"
            display(x_adv,path)
            image = Image.open(path)
            inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
            x_new = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
            h_new = clip(x_new)
            print(criterion(h_new, h_harm))

    path = "adv_random_0.01.png"
    display(x_adv,path)






