import torch
import numpy as np
from inverse.utils import *
from tqdm import tqdm

class Algorithm:
    def __init__(self, generator, config, get_latent, gen_forward):
        self.generator = generator
        self.config = config
        self.get_latent = get_latent
        self.gen_forward = gen_forward

    def invert(self):
        raise NotImplementedError()
                
    def get_default_config(self):
        raise NotImplementedError()
    
    def _get_forward_operator(self, x):
        if self.config['forward'] == 'inpainting':
            return Inpainter(get_inpainting_mask(x))
        else:
            raise NotImplementedError()
    
    def _listify(self, objects):
        output_list = []
        for curr_item in objects:
            if isinstance(curr_item, list):
                output_list += curr_item
            else:
                output_list.append(curr_item)
        return [x for x in output_list if x is not None]

class PGD(Algorithm):
    '''
        Paper: Solving linear inverse problems using GAN priors
    '''    
    def get_default_config(self):
        return {
            'forward': 'inpainting',
            'device': 'cuda',
            'lr': 1e-3,
            'batch_size': 1,
            'image_shape': [1024, 1024, 3],
            'image_steps': 100,
            'latent_steps': 50,
        }

    def invert(self, x):        
        self.forward = self._get_forward_operator(x)
        # w has the same shape with x
        w = torch.zeros((self.config['batch_size'],) + tuple(self.config['image_shape'])).to(self.config['device']) 
        w.requires_grad = True
        image_optimizer = torch.optim.Adam([w], lr=self.config['lr'])

        image_pbar = tqdm(range(self.config['image_steps']))
        
        for image_t in image_pbar:
            w_f = self.forward(w)
            image_loss = F.mse_loss(w_f, x).mean()
            image_loss.backward()
            
            latent = self.get_latent(with_grad=True)
            latent_optimizer = torch.optim.Adam(self._listify(latent), lr=self.config['lr'])
            latent_pbar = tqdm(range(self.config['latent_steps']))
            
            for latent_t in latent_pbar:
                latent_loss = F.mse_loss(self.forward(self.gen_forward(self.generator, latent)), x).mean()
                latent_loss.backward()
            w.data = self.gen_forward(self.generator, latent).data
        return w

    
        
    




    
