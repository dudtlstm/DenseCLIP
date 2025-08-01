import os
import sys
import torch
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

# custom module 경로 등록
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from denseclip.untils import tokenize

def extract_text_embeddings(config_path, checkpoint_path, save_path, device='cuda:0'):
    # Load config
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None  # No need for ImageNet pretrain

    cfg.model.class_names = [
        'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
        'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
        'portal_vein_and_splenic_vein', 'pancreas',
        'adrenal_gland_right', 'adrenal_gland_left'
    ]

    # Build model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()
    model.to(device)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    print(f'[INFO] Loaded checkpoint from: {checkpoint_path}')

    # Prepare tokens
    context_length = model.context_length
    class_names = cfg.model.class_names
    texts = torch.cat([tokenize(c, context_length=context_length) for c in class_names]).to(device)  # (K, context_len)

    # Prepare context prompt
    context_prompt = model.contexts.to(device)  # (1, N_ctx, dim)
    text_embeddings = model.text_encoder(texts, context_prompt)  # (1, K, dim)

    # If context_decoder is used
    if hasattr(model, 'context_decoder'):
        dummy_visual_context = torch.zeros((1, 1, text_embeddings.shape[-1]), device=device)  # dummy visual context
        text_diff = model.context_decoder(text_embeddings, dummy_visual_context)
        text_embeddings = text_embeddings + model.gamma.to(device) * text_diff  # (1, K, dim)

    final_embeddings = text_embeddings[0].detach().cpu()  # (K, dim)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(final_embeddings, save_path)
    print(f'[INFO] Text embeddings saved to: {save_path}')


if __name__ == '__main__':
    config_path = 'configs/denseclip_fpn_res50_512x512_80k_btcv.py'
    checkpoint_path = 'work_dirs/denseclip_fpn_res50_512x512_80k_btcv/latest.pth'
    save_path = 'work_dirs/btc_text_embeddings.pth'

    extract_text_embeddings(config_path, checkpoint_path, save_path)