import os
import torch
import argparse
import csv
import numpy as np
from PIL import Image
from torchvision import models
from tqdm import tqdm

from utils.img_utils import load_image, save_image
from utils.torch_utils import numpy_to_variable, variable_to_numpy
from attack.feature_map_attack_visual import feature_map_attack


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser(
                    prog='privacy protection',
                    description='for the same task, protect one model, attack others',
                    epilog='Text at the bottom of help')

parser.add_argument('--ori_path', type = str, default = '/home/wchai01/Data/selected_imagen')
parser.add_argument('--store_path', type = str, default = '/home/wchai01/Data/generated_imagen')
parser.add_argument('--exp_name', type = str, default = 'testtest')
parser.add_argument('--protect_model_name', type = str, default = 'vgg11')
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--method', type = str, default = 'fm')
parser.add_argument('--steps', type = int, default = 100)
parser.add_argument('--wlambda', type = int, default = 10)
parser.add_argument('--eps', type = float, default = 16/255)
parser.add_argument('--step_size', type = float, default = 4/255)
parser.add_argument('--decay_factor', type = float, default = 0.5)
parser.add_argument('--random_start', type = bool, default = False)
parser.add_argument('--attack_layers', type = list_of_ints, default = [4])


args = parser.parse_args()

# store experiment info
exp_info = ['exp_name', 'protected', 'steps', 'eps', 'step_size', 'description', 'method']
exp_description = [args.exp_name, args.protect_model_name, args.steps, args.eps, args.step_size,'test', args.method]
with open(os.path.join(args.store_path, 'exp_info.csv'),'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(exp_info)
    csvwriter.writerow(exp_description)
# store end

generated_folder = os.path.join(args.store_path,args.exp_name)
if not os.path.exists(generated_folder):
    os.mkdir(generated_folder)

protect_model = getattr(models,args.protect_model_name)(pretrained = True).to(args.device).eval()
attack_layers = args.attack_layers
print(attack_layers)
if args.method == 'fm':
    attack = feature_map_attack(
        model = protect_model,
        epsilon = args.eps,
        step_size = args.step_size,
        device = args.device,
        steps = args.steps,
        decay_factor = args.decay_factor,
        random_start = args.random_start
    )

img_name_list = os.listdir(args.ori_path)
for img_name in tqdm(img_name_list):
    generated_img_store_path = os.path.join(generated_folder,img_name.replace('JPEG','png'))
    image_path = os.path.join(args.ori_path, img_name)
    img = load_image(shape = (224,224),data_format="channels_first", abs_path=True, fpath=image_path)
    img_var = numpy_to_variable(img,args.device)
    generated, ori, small, large = attack(img_var.cpu(),attack_layers)
    generated_np = variable_to_numpy(generated).squeeze(0)
    generated_pil = Image.fromarray(np.transpose((generated_np * 255).astype(np.uint8), (1, 2, 0)))
    generated_pil.save(generated_img_store_path)
