import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import timm
import cv2
from pytorch_grad_cam import GradCAM
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def prepare_image(path, size=224):
    img = Image.open(path).convert('RGB').resize((size, size))
    raw = np.array(img)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    tensor = preprocess(img).unsqueeze(0)
    return raw, tensor


def load_model(path, num_classes, name, device):
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    chk = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(chk)
    model.to(device); model.eval()
    return model


def blend_heatmap(raw, mask, alpha=0.4):
    h, w, _ = raw.shape
    mask_resized = cv2.resize(mask, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(mask_resized*255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = raw.astype(np.float32) / 255.0
    vis = heatmap * alpha + img * (1 - alpha)
    return np.uint8(vis * 255)


def legrad_saliency(model, tensor, raw, layer_module, target_idx=None):
    model.zero_grad()
    A = None
    dA = None

    if hasattr(layer_module, 'attn'):
        attn_mod = layer_module.attn
    else:
        attn_mod = model.layers[-1].blocks[-1].attn

    def forward_attn_hook(mod, inp, out):
        nonlocal A
        A = out
    h1 = attn_mod.softmax.register_forward_hook(forward_attn_hook)

    out = model(tensor)
    if target_idx is None:
        target_idx = out.argmax(dim=1).item()

    def save_grad(grad):
        nonlocal dA
        dA = grad
    h2 = A.register_hook(save_grad)

    loss = out[0, target_idx]
    loss.backward()

    h1.remove()

    A_mean  = A.mean(dim=1)
    dA_mean = dA.mean(dim=1)
    sal     = (A_mean * dA_mean).sum(dim=2)
    sal     = F.relu(sal)[0]
    sal     = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    N       = sal.size(0)
    sz      = int(np.sqrt(N))
    sal_map = sal.reshape(sz, sz).cpu().detach().numpy()

    return blend_heatmap(raw, sal_map)

def gradcam_vis(model, tensor, raw, layer_module):
    cam = GradCAM(model=model, target_layers=[layer_module])
    mask = cam(input_tensor=tensor)[0]
    return blend_heatmap(raw, mask)

def attention_vis(model, tensor, raw, stage, block, attn_type):
    attn_w = None
    def hook(mod, inp, out):
        nonlocal attn_w
        if attn_type == 'attn':
            attn_w = out[1]
        elif attn_type == 'attn_drop':
            attn_w = mod.attn_drop(mod.get_attn())
    blk = model.layers[stage].blocks[block].attn
    h = blk.register_forward_hook(hook)
    _ = model(tensor)
    h.remove()
    w = attn_w[0].mean(0).cpu().detach().numpy()
    N = w.shape[-1]
    sz = int(np.sqrt(N))
    mask = w[0, 1:].reshape(sz, sz) if w.shape[0]>1 else w.reshape(sz, sz)
    return blend_heatmap(raw, mask)


def shap_vis(model, bg, tensor, raw):
    expl = shap.GradientExplainer(model, bg)
    sv = expl.shap_values(tensor)
    sv = [np.transpose(x, (0,2,3,1)) for x in sv]
    shap.image_plot(sv, raw[None]/255.0)


def lime_vis(model, raw, transform_norm, device):
    def batch_pred(imgs):
        t = torch.stack([transform_norm(Image.fromarray(img)) for img in imgs]).to(device)
        return model(t).detach().cpu().numpy()
    expl = lime_image.LimeImageExplainer()
    exp = expl.explain_instance(raw, batch_pred, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    return mark_boundaries(temp/255.0, mask)


def select_layer(model, name):
    if name == 'patch_embed': return model.patch_embed.proj
    if name.startswith('layers'):
        idx = int(name[-1])
        return model.layers[idx].blocks[-1]
    if name == 'norm':
        return model.norm if hasattr(model,'norm') else model.layers[-1].blocks[-1].norm2
    raise ValueError(f'Unknown layer {name}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['gradcam','legrad','attn','shap','lime'], required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--image_path', required=True)
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--model_name', default='swin_base_patch4_window7_224')
    p.add_argument('--output_dir', default='results')
    p.add_argument('--layer',
                   choices=['patch_embed','layers0','layers1','layers2','layers3','norm'],
                   default='layers3')
    p.add_argument('--attn_stage', type=int, default=0)
    p.add_argument('--attn_block', type=int, default=0)
    p.add_argument('--attn_type', choices=['attn','attn_drop'], default='attn')
    p.add_argument('--background_dir')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, args.num_classes, args.model_name, device)
    raw, tensor = prepare_image(args.image_path)
    tensor = tensor.to(device)

    out = model(tensor)
    probs = F.softmax(out, 1)
    idx, pr = probs.argmax(1).item(), probs.max().item()

    class_names = {
        0: 'DCIS',
        1: 'Endothelial',
        2: 'Invasive_Tumor',
        3: 'Myoepi_ACTA2',
        4: 'Myoepi_KRT15'
    }
    
    print(f'Predicted: {class_names[idx]}, probability {pr:.4f}')
    
    with open(os.path.join(args.output_dir, 'prediction_result.txt'), 'w') as f:
        f.write(f'Predicted: {class_names[idx]}, probability {pr:.4f}\n')

    if args.mode in ['gradcam','legrad']:
        module = select_layer(model, args.layer)
        if args.mode == 'gradcam':
            vis, fname = gradcam_vis(model, tensor, raw), 'gradcam.png'
        else:
            vis, fname = legrad_saliency(model, tensor, raw, module), 'legrad.png'
        plt.imsave(os.path.join(args.output_dir, fname), vis)

    elif args.mode == 'attn':
        vis = attention_vis(model, tensor, raw, args.attn_stage, args.attn_block, args.attn_type)
        plt.imsave(os.path.join(args.output_dir, 'attn.png'), vis)

    #don't use this!!
    elif args.mode == 'shap':
        from glob import glob
        files = glob(os.path.join(args.background_dir, '*.*'))[:10]
        bgs = [prepare_image(f, tensor.shape[-1])[1] for f in files]
        bg = torch.cat(bgs,0).to(device)
        shap_vis(model, bg, tensor, raw)
        plt.savefig(os.path.join(args.output_dir, 'shap.png')); plt.clf()

    elif args.mode == 'lime':
        transform_norm = transforms.Compose([
            transforms.Resize(raw.shape[:2]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        vis = lime_vis(model, raw, transform_norm, device)
        plt.imsave(os.path.join(args.output_dir, 'lime.png'), vis)

    print('Saved to', args.output_dir)

if __name__=='__main__':
    main()
