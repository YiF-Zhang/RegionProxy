from collections import OrderedDict


def convert_vit(ckpt):

    new_ckpt = OrderedDict()
    is_hybrid = any(k.startswith('patch_embed.backbone') for k in ckpt)

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('patch_embed'):
            if 'proj' in k and not is_hybrid:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


def old_convert_vit(ckpt, model=None):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('patch_embed'):
            k = k.replace('backbone.', '')
            if model is not None and hasattr(model, 'stem'):
                new_k = k.replace('patch_embed', 'stem')
            else:
                if 'proj' in k:
                    new_k = k.replace('proj', 'projection')
                else:
                    new_k = k
        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt