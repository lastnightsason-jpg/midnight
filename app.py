import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
import tempfile
import timm

# ----------------- CONFIG -----------------
MODEL_FILES = {
    "ResNet50": "https://huggingface.co/sason2004/booldnigth/resolve/main/ResNet50.pt",
    "DenseNet121": "https://huggingface.co/sason2004/booldnigth/resolve/main/DenseNet121.pt",
    "MobileNetV3": "https://huggingface.co/sason2004/booldnigth/resolve/main/MobileNetV3.pt",
    "EfficientNet": "https://huggingface.co/sason2004/booldnigth/resolve/main/EfficientNet.pt",
    "vit_base_patch16_224": "https://huggingface.co/sason2004/booldnigth/resolve/main/vit_base_patch16_224.pt",
}

CLASS_NAMES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
IMG_ROOT = "archive/Datasets"

# ----------------- UTILS -----------------
def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model

def get_first_conv_layer(model):
    model = unwrap_model(model)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None

def get_last_conv_layer(model, model_name):
    model = unwrap_model(model)
    if "resnet" in model_name.lower():
        return model.layer4[-1].conv2
    elif "densenet" in model_name.lower():
        return model.features[-1]
    elif "mobilenet" in model_name.lower():
        return model.blocks[-1]
    elif "efficientnet" in model_name.lower():
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv
    else:
        return None

def get_transform(model_name):
    size = 224 if "vit" in model_name.lower() else 128
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

# ----------------- MODEL LOADING -----------------
def load_model(model_name, model_path):
    """โหลด PyTorch model จาก URL/local รองรับ state_dict/checkpoint/full model"""
    # เลือก class ของโมเดล
    if "resnet" in model_name.lower():
        from torchvision.models import resnet50
        model_class = lambda: resnet50(num_classes=len(CLASS_NAMES))
    elif "densenet" in model_name.lower():
        from torchvision.models import densenet121
        model_class = lambda: densenet121(num_classes=len(CLASS_NAMES))
    elif "mobilenet" in model_name.lower():
        from torchvision.models import mobilenet_v3_large
        model_class = lambda: mobilenet_v3_large(num_classes=len(CLASS_NAMES))
    elif "efficientnet" in model_name.lower():
        from torchvision.models import efficientnet_b0
        model_class = lambda: efficientnet_b0(num_classes=len(CLASS_NAMES))
    elif "vit" in model_name.lower():
        model_class = lambda: timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES)
        )
    else:
        raise ValueError("Unknown model type")

    # โหลดไฟล์จาก URL หรือ local
    if str(model_path).startswith("http"):
        r = requests.get(model_path)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(r.content)
            tmp_path = f.name
    else:
        tmp_path = model_path

    # โหลด state dict หรือ full model
    state = torch.load(tmp_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model = model_class()
        model.load_state_dict(state["state_dict"], strict=False)
    elif isinstance(state, dict) and any("weight" in k or "bias" in k for k in state.keys()):
        model = model_class()
        model.load_state_dict(state, strict=False)
    else:
        model = state  # full model

    model.eval()
    return model

# ----------------- Grad-CAM -----------------
def generate_gradcam(model, img_tensor, target_layer, conv_dtype):
    img_tensor = img_tensor.to(dtype=conv_dtype).requires_grad_()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1,2), keepdim=True)
    cam = (weights*act).sum(0)
    cam = torch.relu(cam)
    cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    cam_img = np.uint8(cam.cpu().numpy()*255)
    cam_img = np.stack([cam_img]*3, axis=2)
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray(cam_img).resize((128,128), resample=PILImage.BILINEAR)

    handle_fwd.remove()
    handle_bwd.remove()
    return cam.cpu().numpy(), cam_img

# ----------------- ViT Attention Rollout -----------------
def vit_attention_rollout(model, img_tensor):
    model.eval()
    attn_weights = []
    hooks = []

    def get_attn(module, input, output):
        attn_weights.append(output[0] if isinstance(output, tuple) else output)

    for block in model.blocks:
        if hasattr(block, "attn"):
            hooks.append(block.attn.register_forward_hook(get_attn))

    with torch.no_grad():
        _ = model(img_tensor)

    for h in hooks:
        h.remove()

    if not attn_weights:
        return None

    attn = attn_weights[0]
    result = torch.eye(attn.shape[-1])
    for att in attn_weights:
        att = att[0].mean(0)
        att = att / att.sum(dim=-1, keepdim=True)
        result = att @ result

    mask = result[0,1:]
    num_patch = int(mask.shape[0]**0.5)
    mask = mask.reshape(num_patch,num_patch).cpu().numpy()
    mask = (mask - mask.min())/(mask.max()-mask.min()+1e-8)
    return mask

# ----------------- STREAMLIT UI -----------------
st.title("White Blood Cell Classifier with Grad-CAM")

# เลือกโมเดล
model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
model_path = MODEL_FILES[model_name]

# โหลดโมเดล **ไม่ใช้ @st.cache_resource** เพื่อป้องกัน crash
model = load_model(model_name, model_path)

# เลือกรูปภาพ
all_images = []
for cls in CLASS_NAMES:
    folder = os.path.join(IMG_ROOT, cls)
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                all_images.append((cls, os.path.join(folder,fname)))
all_images.sort()

img_options = ["[Upload Your Own]"] + [f"{cls}/{os.path.basename(path)}" for cls,path in all_images]
img_idx = st.selectbox("Select an image", range(len(img_options)), format_func=lambda i: img_options[i])

image = None
if img_idx == 0:
    uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    img_path = all_images[img_idx-1][1]
    if os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")

if image is None:
    st.stop()

st.image(image, caption="Selected Image", use_container_width=True)

# ----------------- PREDICTION + GRAD-CAM -----------------
if st.button("Predict & Show Grad-CAM"):
    transform = get_transform(model_name)
    img_tensor = transform(image).unsqueeze(0)

    size = 224 if "vit" in model_name.lower() else 128
    unwrapped_model = unwrap_model(model)
    first_conv = get_first_conv_layer(unwrapped_model)

    if "vit" not in model_name.lower() and first_conv is not None:
        conv_dtype = first_conv.weight.dtype
        img_tensor = img_tensor.to(dtype=conv_dtype)

    with torch.no_grad():
        outputs = unwrapped_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted = np.argmax(probabilities)
        pred_class = CLASS_NAMES[predicted]
        confidence = probabilities[predicted]

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f})")
    st.dataframe({"Class": CLASS_NAMES, "Probability": [f"{p:.4f}" for p in probabilities]})

    if "vit" in model_name.lower():
        attn_map = vit_attention_rollout(unwrapped_model, img_tensor)
        if attn_map is not None:
            img_np = np.array(image.resize((size,size))).astype(np.float32)/255.0
            attn_color = plt.get_cmap('jet')(attn_map)[..., :3]
            overlay = np.clip(0.5*img_np + 0.5*attn_color, 0,1)
            st.image([img_np, attn_color, overlay], caption=["Input","Attention","Overlay"], use_column_width=True)
    else:
        last_conv = get_last_conv_layer(unwrapped_model, model_name)
        if last_conv is not None:
            cam_np, cam_img = generate_gradcam(unwrapped_model, img_tensor, last_conv, conv_dtype)
            img_np = np.array(image.resize((size,size))).astype(np.float32)/255.0
            heatmap_img = plt.get_cmap('jet')(cam_np)[..., :3]
            overlay = np.clip(0.5*img_np + 0.5*heatmap_img, 0,1)
            st.image([img_np, heatmap_img, overlay], caption=["Input","Grad-CAM","Overlay"], use_column_width=True)
