import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
import tempfile

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

# ----------------- FUNCTIONS -----------------
def vit_attention_rollout(model, img_tensor):
    model.eval()
    attn_weights = []
    hooks = []

    def get_attn(module, input, output):
        if isinstance(output, tuple):
            attn_weights.append(output[1] if len(output) > 1 else output[0])
        else:
            attn_weights.append(output)

    for i, block in enumerate(model.blocks):
        found = False
        if hasattr(block.attn, "attn_drop"):
            hooks.append(block.attn.attn_drop.register_forward_hook(get_attn))
            found = True
        elif hasattr(block.attn, "proj_drop"):
            hooks.append(block.attn.proj_drop.register_forward_hook(get_attn))
            found = True
        elif hasattr(block, "attn"):
            hooks.append(block.attn.register_forward_hook(get_attn))
            found = True
        if not found:
            st.write(f"Block {i} attn ไม่เจอ hook ที่รองรับ")

    with torch.no_grad():
        _ = model(img_tensor)

    for h in hooks:
        h.remove()

    if not attn_weights:
        st.warning("ไม่สามารถดึง attention weights จาก ViT ได้")
        return None

    attn = attn_weights[0]
    if isinstance(attn, tuple):
        attn = attn[0]

    if attn.dim() == 4:
        result = torch.eye(attn.shape[-1])
        for attn in attn_weights:
            if isinstance(attn, tuple):
                attn = attn[0]
            attn = attn[0].mean(0)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = attn @ result
        mask = result[0, 1:]
        num_patch = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(num_patch, num_patch).cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() + 1e-8)
        return mask
    else:
        st.warning("ไม่สามารถตีความ attention weights ได้ (shape ไม่ถูกต้อง)")
        return None

def get_transform(model_name):
    size = 224 if "vit" in model_name.lower() else 128
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

@st.cache_resource
def load_model(model_name, model_path):
    if "resnet" in model_name.lower():
        from torchvision.models import resnet50
        model = resnet50(num_classes=len(CLASS_NAMES))
    elif "densenet" in model_name.lower():
        from torchvision.models import densenet121
        model = densenet121(num_classes=len(CLASS_NAMES))
    elif "mobilenet" in model_name.lower():
        from torchvision.models import mobilenet_v3_large
        model = mobilenet_v3_large(num_classes=len(CLASS_NAMES))
    elif "efficientnet" in model_name.lower():
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(num_classes=len(CLASS_NAMES))
    elif "vit" in model_name.lower():
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES))
    else:
        raise ValueError("Unknown model type")

    if model_path.startswith("http"):
        r = requests.get(model_path)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(r.content)
            tmp_path = f.name
        state = torch.load(tmp_path, map_location="cpu")
    else:
        state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state

    model.eval()
    return model

def unwrap_model(model):
    for attr in ['model', 'module', '_forward_module']:
        if hasattr(model, attr):
            return unwrap_model(getattr(model, attr))
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
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()
    cam_img = np.uint8(cam_np * 255)
    cam_img = np.stack([cam_img]*3, axis=2)
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray(cam_img).resize((128, 128), resample=PILImage.BILINEAR)

    handle_fwd.remove()
    handle_bwd.remove()
    return cam_np, cam_img

def load_model(model_name, model_path, model_class, *args, **kwargs):
    """
    Load a PyTorch model from file or URL, no lightning.fabric, no weights_only.
    """
    import requests

    # ตรวจสอบว่าเป็น URL หรือไฟล์ local
    if str(model_path).startswith("http"):
        r = requests.get(model_path)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(r.content)
            tmp_path = f.name
    else:
        tmp_path = model_path

    # โหลด state dict หรือทั้งโมเดล
    state = torch.load(tmp_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model = model_class(*args, **kwargs)
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model = model_class(*args, **kwargs)
        model.load_state_dict(state)
    else:
        model = state  # กรณี save ทั้งโมเดล
    model.eval()
    return model

# ----------------- STREAMLIT UI -----------------
st.title("White Blood Cell Classifier with Grad-CAM")

# เลือกโมเดล
model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
model_path = MODEL_FILES[model_name]
model = load_model(model_name, model_path, MyModelClass)

# เลือกรูปจาก archive หรืออัปโหลด
all_images = []
for cls in CLASS_NAMES:
    folder = os.path.join(IMG_ROOT, cls)
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append((cls, os.path.join(folder, fname)))
all_images.sort()

img_options = [f"{cls}/{os.path.basename(path)}" for cls, path in all_images]
img_options = ["[อัปโหลดรูปภาพของคุณเอง]"] + img_options

img_idx = st.selectbox("Select an image from archive or upload",
                       range(len(img_options)),
                       format_func=lambda i: img_options[i])

image = None
if img_idx == 0:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"ไม่สามารถเปิดไฟล์รูปได้: {e}")
else:
    img_path = all_images[img_idx-1][1]
    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            st.error(f"ไม่สามารถเปิดไฟล์รูปจาก archive ได้: {e}")
    else:
        st.error(f"ไฟล์รูปไม่พบ: {img_path}")

if image is None or not isinstance(image, Image.Image):
    st.stop()

st.image(image, caption="Selected image", use_container_width=True)

# ----------------- PREDICTION + GRAD-CAM -----------------
if st.button("Predict & Show Grad-CAM"):
    transform = get_transform(model_name)
    img_tensor = transform(image).unsqueeze(0)

    size = 224 if "vit" in model_name.lower() else 128
    unwrapped_model = unwrap_model(model)
    first_conv = get_first_conv_layer(unwrapped_model)

    if "vit" not in model_name.lower():
        if first_conv is None:
            st.error("Cannot find first Conv2d layer in model.")
            st.stop()
        conv_dtype = first_conv.weight.dtype
        img_tensor = img_tensor.to(dtype=conv_dtype)

    with torch.no_grad():
        outputs = unwrapped_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted = np.argmax(probabilities)
        pred_class = CLASS_NAMES[predicted]
        confidence = probabilities[predicted]

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f})")
    st.subheader("Class Probabilities")
    st.dataframe({"Class": CLASS_NAMES, "Probability": [f"{p:.4f}" for p in probabilities]})
    st.bar_chart({cls: prob for cls, prob in zip(CLASS_NAMES, probabilities)})

    # Grad-CAM / Attention Map
    if "vit" in model_name.lower():
        attn_map = vit_attention_rollout(unwrapped_model, img_tensor)
        if attn_map is not None:
            img_np = np.array(image.resize((size, size))).astype(np.float32)/255.0
            attn_map_resized = np.array(Image.fromarray(np.uint8(attn_map*255)).resize((size, size), resample=Image.BILINEAR))/255.0
            attn_color = plt.get_cmap('jet')(attn_map_resized)[..., :3]
            overlay = np.clip(0.5*img_np + 0.5*attn_color, 0, 1)
            st.subheader("ViT Attention Map Visualization")
            col1, col2, col3 = st.columns(3)
            with col1: st.image(img_np, caption="Input Image", use_container_width=True)
            with col2: st.image(attn_color, caption="Attention Map", use_container_width=True)
            with col3: st.image(overlay, caption="Overlay", use_container_width=True)
    else:
        last_conv = get_last_conv_layer(unwrapped_model, model_name)
        if last_conv is not None:
            cam_np, cam_img = generate_gradcam(unwrapped_model, img_tensor, last_conv, conv_dtype)
            img_np = np.array(image.resize((size, size))).astype(np.float32)/255.0
            heatmap = (cam_np - cam_np.min())/(cam_np.max() - cam_np.min() + 1e-8)
            heatmap_img = plt.get_cmap('jet')(heatmap)[..., :3]
            overlay = np.clip(0.5*img_np + 0.5*heatmap_img, 0, 1)
            st.subheader("Grad-CAM Visualization")
            col1, col2, col3 = st.columns(3)
            with col1: st.image(img_np, caption="Input Image", use_container_width=True)
            with col2: st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
            with col3: st.image(overlay, caption="Overlay", use_container_width=True)
        else:
            st.warning("Grad-CAM is not supported for this model.")
