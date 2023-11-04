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

    @torch.no_grad()
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
    # mm_vision_tower='openai/clip-vit-large-patch14'
    mm_vision_tower='openai/clip-vit-large-patch14-336'
    # openai/clip-vit-large-patch14-336
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



import argparse


parser = argparse.ArgumentParser(description='')


parser.add_argument('--num_samples', type=int)

args = parser.parse_args()





# inputs shape torch.Size([1, 3, 224, 224])
# embedding shape torch.Size([1, 256, 1024])


# load a harmful images
image = Image.open('bomb_336.png')
inputs = clipProcessor(images=image, return_tensors="pt", padding=True)

# text harmful 
# inputs = clipProcessor(text=["Grenade Bomb"],return_tensors="pt", padding=True)
# print(inputs)
x_harm = torch.tensor(inputs['pixel_values'], dtype=torch.float32)


# load a adv images (have their own meanings)

image = Image.open("/home/changsheng/adv_vlm/cat_336.png")
inputs = clipProcessor( images=image, return_tensors="pt", padding=True)
x_adv = torch.tensor((inputs['pixel_values']).clone().detach(), dtype=torch.float32)
x_ori = torch.tensor((inputs['pixel_values']), dtype=torch.float32).clone()
'''
image = Image.open("Cat_sticker.png")
inputs = clipProcessor(text=["Grenade Bomb"],images=image, return_tensors="pt", padding=True)
x_ori = torch.tensor(inputs['pixel_values'].clone().detach(), dtype=torch.float32)
'''
# the patch you can select: stationary and moving

print('------------------------- Start Running ----------')

# we first choose the stationary one 
patch_type = 'stationary'
patch_size = 120
start_x_y = (216,0)
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


type_attack = 'random_S'
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
            path = "adv_random_0.01_336.png"
            display(x_adv,path)
            image = Image.open(path)
            inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
            x_new = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
            h_new = clip(x_new)
            print(criterion(h_new, h_harm))

    path = "adv_random_0.01_336.png"
    display(x_adv,path)


elif type_attack == 'patch':
    
    print('------------ Patch ------------')
    
    # device 
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    
    # load the model and tensor on device
    x_ori = torch.randn(1, 3, 336, 336, requires_grad=True)
    clip.to(device)
    x_adv.to(device)
    x_harm.to(device)
    x_ori.to(device)
    
    # Load a optimize(Adam,..)
    
    # init a img x
    x = (1-mask)*x_adv + mask*x_ori
    # x = x_ori
    x.to(device)
    x = x.detach()
    x.requires_grad_()

    # load the para 
    learning_rate = 0.1
    stop_para = 0.1
    epoch_counter = 0
    loss_now = 9999
    
    optimizer = optim.Adam([x], lr=learning_rate)  
    criterion = nn.MSELoss() # L2 Loss
    
    while loss_now > stop_para:
        
        optimizer.zero_grad()  
        
        h_adv = clip(x)
        h_harm = clip(x_harm)
        loss = criterion(h_adv, h_harm)
        loss_now = loss.item()
        loss.backward()
        optimizer.step()
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        stddev = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        normalized_image = x.data
        restored_image = normalized_image * stddev.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        restored_image = restored_image.clamp(0, 1)
        x.data = (restored_image - mean.view(1, 3, 1, 1)) / stddev.view(1, 3, 1, 1)
        x.data  = (1-mask)*x_adv + mask*x.data
        
        print(f'Epoch [{epoch_counter}], Loss: {loss.item()}')
        epoch_counter+=1
        if epoch_counter % 100 == 0:
            path = "adv_patch_0.1_336_120_nineb.png"
            display(x,path)
            image = Image.open(path)
            inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
            x_new = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
            h_new = clip(x_new)
            print(criterion(h_new, h_harm))
            print(path)
    
    path = "adv_patch_0.1_336_120_nineb.png"
    display(x,path)
    
    
elif type_attack == 'patch_only':
    

    shape = (1, 3, patch_size, patch_size)
    patch = torch.randn(shape,requires_grad=True)
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    
    # load the model and tensor on device
    # x_ori = torch.randn(1, 3, 336, 336, requires_grad=True)
    clip.to(device)
    x_adv.to(device)
    x_harm.to(device)
    patch.to(device)
    # Load a optimize(Adam,..)
    

    # load the para 
    learning_rate = 0.1
    stop_para = 0.1
    epoch_counter = 0
    loss_now = 9999
    
    patch = patch.detach()
    patch.requires_grad_()
    
    optimizer = optim.Adam([patch.requires_grad_()], lr=learning_rate)  
    criterion = nn.MSELoss() # L2 Loss
    
    x_adv[:, :, x_position:x_position + patch_size, y_position:y_position + patch_size].data = patch.data

    while loss_now > stop_para:
        # print("y_non_leaf is a leaf tensor:", x_adv.is_leaf)
        
        x_adv[:, :, x_position:x_position + patch_size, y_position:y_position + patch_size].data = patch.data
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        stddev = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        normalized_image = x_adv.data
        restored_image = normalized_image * stddev.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        restored_image = restored_image.clamp(0, 1)
        x_adv.data = (restored_image - mean.view(1, 3, 1, 1)) / stddev.view(1, 3, 1, 1)
        patch.requires_grad_()
        patch.data = x_adv[:, :, x_position:x_position + patch_size, y_position:y_position + patch_size].data

        x_adv[:, :, x_position:x_position + patch_size, y_position:y_position + patch_size].data = patch.data
        
        optimizer.zero_grad()  
        h_adv = clip(x_adv)
        h_harm = clip(x_harm)
        loss = criterion(h_adv, h_harm)
        loss_now = loss.item()
        loss.backward()
        print(patch.grad)
        optimizer.step()
        
        print(f'Epoch [{epoch_counter}], Loss: {loss.item()}')
        epoch_counter+=1
        if epoch_counter % 100 == 0:
            path = "patch_0.1_336_120_sticker.png"
            display(x_adv,path)
            image = Image.open(path)
            inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
            x_new = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
            h_new = clip(x_new)
            print(criterion(h_new, h_harm))
            print(path)
    
    path = "patch_0.1_336_120_sticker.png"
    display(x_adv,path)
    
elif type_attack == 'test_1':
    # mask some embedding and find the loss diff
    
    
    image = Image.open('bomb_336.png')
    inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
    x_harm = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
    h_harm = clip(x_harm)
    
    image = Image.open('lp_0.1_adv_random_0.01_336.png')
    inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
    x_adv = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
    h_adv = clip(x_adv)
    criterion = nn.MSELoss()
    print(criterion(h_adv, h_harm))
    
    # 
    image = x_adv.clone()
    patch_size = 14
    num_patches = 576

    _, _, height, width = image.size()

    assert height % patch_size == 0 and width % patch_size == 0

    num_rows = height // patch_size
    num_cols = width // patch_size
    loss_list = []
    Loss_dict = []
    for i in range(num_rows):
        for j in range(num_cols):
            print((i,j))
            row_start, row_end = i * patch_size, (i + 1) * patch_size
            col_start, col_end = j * patch_size, (j + 1) * patch_size
            
            image[:, :, row_start:row_end, col_start:col_end] = 0
            
            # path = "mask_img.png"
            # display(image,path)
            h_adv = clip(image)
            criterion = nn.MSELoss()

            # print(criterion(h_adv, h_harm))
            Loss = criterion(h_adv, h_harm).item()
            loss_list.append(Loss)
            Loss_dict.append((Loss,(i,j)))
            image = x_adv.clone()
            

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    # plt.plot(range(len(loss_list)), loss_list)
    plt.scatter(range(len(loss_list)), loss_list, marker='o', s=10)
    plt.xlabel('Index')
    plt.ylabel('Loss')

    plt.savefig('plot_patch_bomb_lp.png')
    
    
    import pickle
    file_name = "patch_bomb_lp_list.pkl"

    with open(file_name, 'wb') as file:
        pickle.dump(Loss_dict, file)
    
    

    
    '''
    image = Image.open('adv_patch_0.1_336_120_oneb.png')
    inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
    x_adv = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
    
    
    h_harm = clip(x_harm)
    h_adv = clip(x_adv)

    print(h_adv.shape)
    
    # h_adv[:, 100 , :] = 0
    
    
    criterion = nn.MSELoss()
    
    print(criterion(h_adv, h_harm))
    
    # Origin  Loss : 0.2701
    
    # Loss 0.3204
    loss_list = []
    
    for i in range(576):
        temp = h_adv.clone()
        temp[:, i , :] = 0
        Loss = criterion(temp, h_harm).item()
        # print(Loss)
        if Loss>0.285:
            print(i)
        loss_list.append(Loss)
    
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    # plt.plot(range(len(loss_list)), loss_list)
    plt.scatter(range(len(loss_list)), loss_list, marker='o', s=10)
    plt.xlabel('Index')
    plt.ylabel('Y Value')

    plt.savefig('plot_adv.png')
    '''
    
elif type_attack == 'adv_lp':
    learning_rate = 0.1
    epsilon = 0.2
    stop_para = 0.3
    epoch_counter = 0
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    clip.to(device)
    x_adv.to(device)
    x_harm.to(device)
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
        restored_ori_image = x_ori * stddev.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        perturbation = torch.clamp((restored_image - restored_ori_image), min=-epsilon, max=epsilon) 
        restored_image = restored_ori_image + perturbation
        restored_image = restored_image.clamp(0, 1)
        x_adv.data = (restored_image - mean.view(1, 3, 1, 1)) / stddev.view(1, 3, 1, 1)
        
        print(f'Epoch [{epoch_counter}], Loss: {loss.item()}')
        epoch_counter += 1
        
        if epoch_counter % 500 == 0:
            path = "lp_0.2_adv_random_0.01_336.png"
            display(x_adv,path)
            image = Image.open(path)
            inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
            x_new = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
            h_new = clip(x_new)
            print(criterion(h_new, h_harm))

    path = "lp_0.2_adv_random_0.01_336.png"
    display(x_adv,path)
    
    
elif type_attack == 'random_S':
    
    image = Image.open('bomb_336.png')
    inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
    x_harm = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
    h_harm = clip(x_harm)
    
    image = Image.open('lp_0.1_adv_random_0.01_336.png')
    inputs = clipProcessor(images=image, return_tensors="pt", padding=True)
    x_adv = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
    h_adv = clip(x_adv)
    criterion = nn.MSELoss()
    print(criterion(h_adv, h_harm))
    
    image = x_adv.clone()
    patch_size = 14
    num_patches = 576

    _, _, height, width = image.size()

    assert height % patch_size == 0 and width % patch_size == 0

    num_rows = height // patch_size
    num_cols = width // patch_size
    loss_list = []
    Loss_dict = []
    criterion = nn.MSELoss()
    import itertools
    import random
    import numpy as np
    # random selct 
    
    all_list = []
    
    coordinates = list(itertools.product(range(24), repeat=2))
    # 576 lenth 
    random_collect = []
    print('---------------------',args.num_samples)
    for i in range(100):
        num_samples = args.num_samples
        random_samples = random.sample(coordinates, num_samples)
        # type [(17, 9), (2, 5)]
        # print(random_samples)
        random_collect.append(random_samples)
        for key in random_samples:
            i = key[0]
            j = key[1]
            row_start, row_end = i * patch_size, (i + 1) * patch_size
            col_start, col_end = j * patch_size, (j + 1) * patch_size
            image[:, :, row_start:row_end, col_start:col_end] = 0
        
        path = "test.png"
        display(image,path)
        
        h_adv = clip(image)
        Loss = criterion(h_adv, h_harm).item()
        loss_list.append(Loss)
        image = x_adv.clone()
    
    
    
    # print(loss_list)
    avg = np.mean(loss_list)
    std = np.std(loss_list)
    
    print(avg,std)
    
    random_collect= [avg,std,random_collect,loss_list]
    
    # print(random_collect)
    
    import pickle
    file_name = f'patch_'+str(num_samples)+'.pkl'

    with open(file_name, 'wb') as file:
        pickle.dump(random_collect, file)
    
    
