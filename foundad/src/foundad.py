import multiprocessing as mp
from typing import Any, Dict, Tuple, Optional, List
import importlib   
import yaml, numpy as np, torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_
from src.datasets.dataset import build_dataloader
import src.dinov2.models.vision_transformer as vit
from transformers import AutoProcessor, SiglipVisionModel, CLIPVisionModel



class LinearProjector(torch.nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class VisionModule(nn.Module):
    def __init__(self, model_name: str, pred_depth: int, pred_emb_dim: int, use_cuda: bool = True, if_pe: bool = True, feat_normed: bool = False, crop_size: int = 512):
        super().__init__()
        self.crop_size = crop_size
        (self.encoder, self.num_patches, self.embed_dim, self.processor, self.projector) = self._build_encoder(model_name)
        self.model_name = model_name

        self.predictor = vit.__dict__["vit_predictor"](num_patches=self.num_patches, embed_dim=self.embed_dim,
                                                         predictor_embed_dim=pred_emb_dim, depth=pred_depth, if_pe=if_pe, feat_normed=feat_normed)
        self._init_predictor(self.predictor)
        self.dropout = nn.Dropout(0.2)
        if use_cuda and torch.cuda.is_available():
            self.cuda()
        self.feat_normed = self.predictor.feat_normed # it depends on the predictor
        print(f"Normed features: {self.feat_normed}")

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)
    
    def target_features(self, images, paths, n_layer=3, use_tensor_feat=False):
        with torch.no_grad():
            return self._extract(images, paths, n_layer=n_layer, use_tensor_feat=use_tensor_feat)

    def context_features(self, images, paths, n_layer=3, use_tensor_feat=False):
        z = self._extract(images, paths, n_layer=n_layer, use_tensor_feat=use_tensor_feat)
        p = self.predictor(self.dropout(z))
        return z, p

    def _build_encoder(self, model: str):

        projector = processor = None
        if model == "dinov2":
            enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval(); num_patches, embed_dim = (self.crop_size // 14) ** 2, enc.embed_dim
        elif model == "dinov3":
            from transformers import AutoModel, AutoProcessor
            enc = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").eval()
            processor = AutoProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
            # Dynamic size for DINOv3 processor
            processor.size = {"height": self.crop_size, "width": self.crop_size}
            num_patches, embed_dim = (self.crop_size // 16) ** 2, 768
        elif model == "dino":
            enc = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval(); num_patches, embed_dim = (self.crop_size // 16) ** 2, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval(); processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512"); num_patches, embed_dim = (self.crop_size // 16) ** 2, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval(); processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16"); num_patches, embed_dim = (self.crop_size // 16) ** 2, 768
        elif model == "dinosiglip":
            from src.vision_backbone.scripts.vit_inference import init_vit_backbone, Config      
            
            config = Config()
            enc = init_vit_backbone(config)

            projector = LinearProjector(2176, 2176).cuda()
            num_patches, embed_dim = 729, 2176
        else:
            raise ValueError(f"Unknown model: {model}")
        if model != 'dinosiglip':
            for p in enc.parameters(): 
                p.requires_grad = False
        return enc, num_patches, embed_dim, processor, projector

    def _extract(self, imgs: torch.Tensor, paths: List[str], n_layer: int = 3, use_tensor_feat: bool = False):
        if self.model_name == "dinov2":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0] # the thrid last block
        elif self.model_name == "dinov3":
            if use_tensor_feat:
                pixel_values = imgs # Assuming already normalized by dataset
            else:
                pil_list = [Image.open(p).convert("RGB") for p in paths]
                proc = self.processor(images=pil_list, return_tensors="pt")
                pixel_values = proc["pixel_values"].to(imgs.device)

            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
                hs = out.hidden_states

            L = len(hs) - 1
            n = max(1, min(n_layer, L))
            # DINOv3 has 1 CLS token + 4 register tokens, so patches start at index 5
            h = hs[-n][:, 5:, :]
        elif self.model_name == "dino":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer)[0][:,1:,:]
        elif self.model_name == "siglip":
            if use_tensor_feat:
                # Optimized for training: Use the provided imgs tensor directly
                # Step 1: Un-normalize from ImageNet stats (Dataset default) back to [0, 1]
                mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
                unnormalized_imgs = imgs * std + mean
                
                # Step 2: Re-normalize for SigLIP [0.5, 0.5, 0.5]
                pixel_values = (unnormalized_imgs - 0.5) / 0.5
            else:
                # Baseline Paper logic: Reload from disk (ignores on-the-fly augmentations/synthesis)
                pil_list = [Image.open(p).convert("RGB") for p in paths]
                proc = self.processor(images=pil_list, return_tensors="pt")
                pixel_values = proc["pixel_values"].to(imgs.device)

            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
                hs = out.hidden_states  # tuple: [embeddings, block1, ..., blockL]; len = L+1

            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, :, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "clip":
            hs = self.encoder(pixel_values=imgs, output_hidden_states=True).hidden_states
            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, 1:, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "dinosiglip":
            feats = [self.encoder.generate(Image.open(p).convert("RGB"))[0] for p in paths]
            h = torch.cat(feats).view(imgs.size(0), 2176, -1).permute(0,2,1)
            h = self.projector(h) if self.projector else h
        else:
            raise NotImplementedError(self.model_name)

        if self.feat_normed:
            h = F.normalize(h, dim=-1)

        return h

    @staticmethod
    def _init_predictor(module):
        for m in module.modules():
            if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=0.02); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)
