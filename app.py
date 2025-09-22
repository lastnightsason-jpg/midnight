import streamlit as st
import torch
import lightning.fabric.wrappers as lf
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import requests
import tempfile
import pandas as pd

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

# ----------------- HELPERS / LOADING -----------------
def load_model_state(model_name, model_path):
    # โหลดไฟล์จาก URL หรือ local
    if str(model_path).startswith("http"):
        r = requests.get(model_path)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(r.content)
            tmp_path = f.name
    else:
        tmp_path = model_path

    # allowlist Fabric module (บางเวอร์ชันของ torch อาจไม่มี API นี้ -> try/except)
    try:
        torch.serialization.add_safe_globals([lf._FabricModule])
    except Exception:
        # ถ้า attribute ไม่มี ให้ข้ามไป (แต่ระวัง security ของไฟล์ pickle)
        pass

    state = torch.load(tmp_path, map_location="cpu", weights_only=False)
    return state

@st.cache_resource
def load_model(model_name, model_path):
    # สร้างโมเดลเปล่า (factory)
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
        import timm
        model_class = lambda: timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    state = load_model_state(model_name, model_path)

    # โหลด state_dict หรือ full pickled model
    if isinstance(state, dict) and "state_dict" in state:
        model = model_class()
        ckpt = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
        model.load_state_dict(ckpt, strict=False)
    elif isinstance(state, dict) and any(("weight" in k or "bias" in k) for k in state.keys()):
        model = model_class()
        model.load_state_dict(state, strict=False)
    else:
        # ถ้าเป็น full pickled model object
        model = state

    model.eval()
    return model

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model

def get_transform(model_name):
    size = 224 if "vit" in model_name.lower() else 128
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def get_first_conv_layer(model):
    model = unwrap_model(model)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None

def get_last_conv_layer(model, model_name):
    model = unwrap_model(model)
    if "resnet" in model_name.lower():
        # last conv before fc
        try:
            return model.layer4[-1].conv2
        except Exception:
            return None
    elif "densenet" in model_name.lower():
        # DenseNet: features[-1] อาจจะเป็น BatchNorm/Relu; หา conv ล่าสุด
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv
    elif "mobilenet" in model_name.lower():
        # MobileNetV3: หา conv ล่าสุด
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv
    elif "efficientnet" in model_name.lower():
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv
    else:
        return None

# ----------------- Grad-CAM / ViT attention -----------------
def generate_gradcam(model, img_tensor, target_layer, conv_dtype):
    img_tensor = img_tensor.to(dtype=conv_dtype).requires_grad_()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple
        gradients.append(grad_out[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    # register_full_backward_hook is supported in newer torch; fallback to register_backward_hook if needed
    try:
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    except Exception:
        handle_bwd = target_layer.register_backward_hook(lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out))

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
    if 'handle_bwd' in locals():
        try:
            handle_bwd.remove()
        except Exception:
            pass
    return cam_np, cam_img

def vit_attention_rollout(model, img_tensor):
    model.eval()
    attn_weights = []
    hooks = []

    def get_attn(module, input, output):
        # output shape differs by implementation; try to capture attention tensor if present
        if isinstance(output, tuple):
            attn_weights.append(output[1] if len(output) > 1 else output[0])
        else:
            attn_weights.append(output)

    # attach hooks to transformer blocks
    if not hasattr(model, "blocks"):
        st.warning("โมเดล ViT ไม่มี attribute 'blocks' ที่คาดหวัง")
        return None

    for i, block in enumerate(model.blocks):
        found = False
        # common spots where attention matrix may appear
        if hasattr(block, "attn") and hasattr(block.attn, "get_attn"):
            # some custom implementations
            try:
                hooks.append(block.attn.register_forward_hook(get_attn))
                found = True
            except Exception:
                pass
        # generic hook on block
        if not found:
            try:
                hooks.append(block.register_forward_hook(get_attn))
                found = True
            except Exception:
                pass
        if not found:
            st.write(f"Block {i} attn ไม่เจอ hook ที่รองรับ")

    with torch.no_grad():
        _ = model(img_tensor)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if not attn_weights:
        st.warning("ไม่สามารถดึง attention weights จาก ViT ได้")
        return None

    # Try to build rollout from collected attn matrices
    attn = attn_weights[0]
    if isinstance(attn, tuple):
        attn = attn[0]

    # Expect attn shape like (batch, heads, tokens, tokens) or similar
    try:
        result = torch.eye(attn.shape[-1])
        for a in attn_weights:
            if isinstance(a, tuple):
                a = a[0]
            a = a[0].mean(0)  # average heads
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
            result = a @ result
        mask = result[0, 1:]  # remove cls token
        num_patch = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(num_patch, num_patch).cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask
    except Exception:
        st.warning("ไม่สามารถตีความ attention weights ได้ (shape ไม่ตรงตามที่คาด)")
        return None

# ----------------- STREAMLIT UI -----------------
st.title("White Blood Cell Classifier with Grad-CAM")

# เลือกโมเดล
model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
model_path = MODEL_FILES[model_name]

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
    try:
        model = load_model(model_name, model_path)
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    transform = get_transform(model_name)
    img_tensor = transform(image).unsqueeze(0)

    size = 224 if "vit" in model_name.lower() else 128
    unwrapped_model = unwrap_model(model)

    # For non-ViT, ensure conv dtype exists
    if "vit" not in model_name.lower():
        first_conv = get_first_conv_layer(unwrapped_model)
        if first_conv is None:
            st.error("Cannot find first Conv2d layer in model.")
            st.stop()
        conv_dtype = first_conv.weight.dtype
        img_tensor = img_tensor.to(dtype=conv_dtype)
    else:
        # for ViT use float32 by default
        conv_dtype = torch.float32
        img_tensor = img_tensor.to(dtype=conv_dtype)

    with torch.no_grad():
        outputs = unwrapped_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted = int(np.argmax(probabilities))
        pred_class = CLASS_NAMES[predicted]
        confidence = float(probabilities[predicted])

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f})")
    st.subheader("Class Probabilities")
    df_probs = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities})
    st.dataframe(df_probs.style.format({"Probability":"{:.4f}"}))
    # bar chart expects dataframe or pd.Series
    st.bar_chart(df_probs.set_index("Class"))

    # Grad-CAM / Attention Map
    if "vit" in model_name.lower():
        attn_map = vit_attention_rollout(unwrapped_model, img_tensor)
        if attn_map is not None:
            img_np = np.array(image.resize((size, size))).astype(np.float32) / 255.0
            attn_map_resized = np.array(
                Image.fromarray(np.uint8(attn_map * 255)).resize((size, size), resample=Image.BILINEAR)
            ) / 255.0
            attn_color = plt.get_cmap('jet')(attn_map_resized)[..., :3]
            overlay = np.clip(0.5 * img_np + 0.5 * attn_color, 0, 1)

            st.subheader("ViT Attention Map Visualization")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_np, caption="Input Image", use_container_width=True)
            with col2:
                st.image(attn_color, caption="Attention Map", use_container_width=True)
            with col3:
                st.image(overlay, caption="Overlay", use_container_width=True)

    else:
        last_conv = get_last_conv_layer(unwrapped_model, model_name)
        if last_conv is not None:
            try:
                cam_np, cam_img = generate_gradcam(unwrapped_model, img_tensor, last_conv, conv_dtype)
            # ✅ resize cam_np ให้เท่ากับ input image size
                cam_resized = np.array(
                    PILImage.fromarray((cam_np * 255).astype(np.uint8)).resize((size, size), resample=PILImage.BILINEAR)
                ).astype(np.float32) / 255.0

                img_np = np.array(image.resize((size, size))).astype(np.float32) / 255.0
                heatmap = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                heatmap_img = plt.get_cmap('jet')(heatmap)[..., :3]
                overlay = np.clip(0.5 * img_np + 0.5 * heatmap_img, 0, 1)

                st.subheader("Grad-CAM Visualization")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img_np, caption="Input Image", use_container_width=True)
                with col2:
                    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                with col3:
                    st.image(overlay, caption="Overlay", use_container_width=True)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดขณะสร้าง Grad-CAM: {e}")
        else:
            st.warning("Grad-CAM is not supported for this model (ไม่พบเลเยอร์ Conv ที่เหมาะสม).")
