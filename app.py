import torch
import tempfile
import requests
import os
import safetensors.torch

def safe_load_state(path):
    """
    พยายามโหลดไฟล์ model จาก path โดยตรวจสอบชนิดไฟล์อัตโนมัติ
    รองรับทั้ง: .pt, .pth, .ckpt, .bin (HuggingFace), .safetensors
    """
    try:
        print(f"➡️ พยายามโหลดไฟล์ด้วย torch.load(): {path}")
        return torch.load(path, map_location="cpu")
    except Exception as e:
        # กรณี safetensors
        if str(path).endswith(".safetensors"):
            print(f"➡️ โหลดด้วย safetensors: {path}")
            return safetensors.torch.load_file(path, device="cpu")
        # ถ้า error อย่างอื่น log ออกมาให้ละเอียด
        size = os.path.getsize(path) if os.path.exists(path) else "N/A"
        raise RuntimeError(
            f"❌ โหลดไฟล์โมเดลไม่สำเร็จ: {type(e).__name__}: {e}\n"
            f"📂 Path: {path}\n"
            f"📏 Size: {size} bytes"
        )


def load_model(model_name, model_path, *args, **kwargs):
    """
    Load a PyTorch model from file or URL (safe for state_dict/checkpoint/full model/safetensors)
    """
    # เลือก class ของโมเดลตามชื่อ
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
        model_class = lambda: timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES)
        )
    else:
        raise ValueError("Unknown model type")

    # ตรวจสอบว่าเป็น URL หรือไฟล์ local
    if str(model_path).startswith("http"):
        r = requests.get(model_path)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(r.content)
            tmp_path = f.name
    else:
        tmp_path = model_path

    # โหลด state
    state = safe_load_state(tmp_path)

    # ตรวจสอบว่า state เป็น dict แบบไหน
    if isinstance(state, dict) and "state_dict" in state:
        print("✅ checkpoint แบบ Lightning: state['state_dict']")
        model = model_class(*args, **kwargs)
        model.load_state_dict(state["state_dict"], strict=False)
    elif isinstance(state, dict) and any("weight" in k or "bias" in k for k in state.keys()):
        print("✅ state_dict ปกติ")
        model = model_class(*args, **kwargs)
        model.load_state_dict(state, strict=False)
    elif isinstance(state, torch.nn.Module):
        print("✅ full model ที่ save ทั้งโมเดล")
        model = state
    else:
        raise RuntimeError(f"❌ state format ไม่รู้จัก: {type(state)}")

    model.eval()
    return model
