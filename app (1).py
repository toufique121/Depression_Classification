"""
Streamlit deployment app for the 3-modality (Text + Numerical + Image) late-fusion
depression-severity classifier, with SHAP / Captum / Grad-CAM explainability and
auto-saving of every student's input+output.

Model loading logic, feature schema, and fusion math are ported directly from the
research notebook (predict_student / explain_student pipeline).
"""

import os
import io
import csv
import json
import shutil
import zipfile
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import streamlit as st

# --------------------------------------------------------------------------- #
# 0. CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Student Mental-Health Screener", layout="wide")

LABELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
LABEL_COLORS = ["#4CAF50", "#CDDC39", "#FFC107", "#FF7043", "#B71C1C"]
IMAGE_SIZE = 224

NUM_COLS = ["Age", "Academic_Stress", "CGPA", "Financial_Stress", "Sleep_Hours", "Screen_Time",
            "Anxiety_Level", "Loneliness_Level", "Family_Support", "Self_Confidence",
            "Appearance_Satisfaction", "Academic_Frustration", "Social_Media_Comparison",
            "Course_Retake_Stress", "Career_Anxiety", "Living_Env_Satisfaction"]
CAT_COLS = ["Gender", "University", "Year_of_Study", "Relationship_Status", "Breakup_Last_6_Months",
            "Part_Time_Job", "Sleep_Quality", "Physical_Activity", "Family_History_Depression",
            "Insomnia_Frequency"]

CAT_OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "University": ["DUET", "DU", "BUET", "RUET", "CUET", "Other"],
    "Year_of_Study": ["1st Year", "2nd Year", "3rd Year", "4th Year"],
    "Relationship_Status": ["Single", "In a relationship", "Married"],
    "Breakup_Last_6_Months": ["Yes", "No"],
    "Part_Time_Job": ["Yes", "No"],
    "Sleep_Quality": ["Poor", "Average", "Good"],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Family_History_Depression": ["Yes", "No"],
    "Insomnia_Frequency": ["Never", "Rarely", "Sometimes", "Often", "Always"],
}

LOG_PATH = "data/predictions_log.csv"
os.makedirs("data", exist_ok=True)

# Model source: set these in Streamlit "Secrets" (Settings -> Secrets) after
# uploading the 3 model files to a Hugging Face model repo. See README.md.
HF_REPO_ID = st.secrets.get("HF_REPO_ID", os.environ.get("HF_REPO_ID", ""))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
TEXT_MODEL_ZIP_NAME = st.secrets.get("TEXT_MODEL_ZIP_NAME", "xlmroberta_mental_state.zip")
NUMERICAL_MODEL_NAME = st.secrets.get("NUMERICAL_MODEL_NAME", "numerical_model_tuned_final.pkl")
IMAGE_MODEL_NAME = st.secrets.get("IMAGE_MODEL_NAME", "final_model_D_full.pth")

LOCAL_EXTRACT_DIR = "model_cache/text_model_extracted"
LOCAL_MODEL_DIR = "model_cache"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. DOWNLOAD + LOAD MODELS (cached, so this runs once per server process)
# --------------------------------------------------------------------------- #
def _hf_download(filename):
    from huggingface_hub import hf_hub_download
    if not HF_REPO_ID:
        raise RuntimeError(
            "HF_REPO_ID is not set. Add it in Streamlit 'Secrets' after uploading "
            "your 3 model files to a Hugging Face model repository."
        )
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename, token=HF_TOKEN,
                            local_dir=LOCAL_MODEL_DIR)


def _find_config_root(base):
    if not os.path.exists(base):
        return None
    for root, _, files in os.walk(base):
        if "config.json" in files:
            return root
    return None


@st.cache_resource(show_spinner="Loading text model (first run only, may take a while)...")
def load_text_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    existing_root = _find_config_root(LOCAL_EXTRACT_DIR)
    if existing_root is None:
        zip_path = _hf_download(TEXT_MODEL_ZIP_NAME)
        os.makedirs(LOCAL_EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(LOCAL_EXTRACT_DIR)
        existing_root = _find_config_root(LOCAL_EXTRACT_DIR)
        if existing_root is None:
            raise FileNotFoundError("config.json not found after extracting the text model zip.")

    tokenizer = AutoTokenizer.from_pretrained(existing_root)
    model = AutoModelForSequenceClassification.from_pretrained(existing_root)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading numerical model...")
def load_numerical_model():
    import joblib
    path = _hf_download(NUMERICAL_MODEL_NAME)
    return joblib.load(path)


def build_image_architecture(state_dict, class_names=None):
    import torchvision.models as tv_models
    if class_names is not None:
        num_classes = len(class_names)
    else:
        fc_key = next((k for k in state_dict.keys() if k.endswith("fc.4.weight")), None)
        if fc_key is None:
            raise NotImplementedError("Could not infer num_classes from checkpoint.")
        num_classes = state_dict[fc_key].shape[0]
    model = tv_models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(num_features, 256), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(256, num_classes),
    )
    return model


@st.cache_resource(show_spinner="Loading image model...")
def load_image_model():
    path = _hf_download(IMAGE_MODEL_NAME)
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        model, class_names = obj, None
    else:
        if isinstance(obj, dict) and "model_state_dict" in obj:
            state_dict, class_names = obj["model_state_dict"], obj.get("class_names")
        else:
            state_dict, class_names = obj, None
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model = build_image_architecture(state_dict, class_names)
        model.load_state_dict(state_dict)

    model.eval()
    model._class_names = class_names
    return model


@st.cache_resource
def get_mtcnn():
    from facenet_pytorch import MTCNN
    return MTCNN(image_size=IMAGE_SIZE, margin=25, keep_all=False, device="cpu")


# --------------------------------------------------------------------------- #
# 2. PREDICTION FUNCTIONS (ported from the notebook's predict_student pipeline)
# --------------------------------------------------------------------------- #
def predict_text(mental_state_text, future_thoughts_text, tokenizer, model, max_len=128):
    text = f"{mental_state_text.strip()} [SEP] {future_thoughts_text.strip()}"
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return F.softmax(logits, dim=1).numpy()[0], inputs


def predict_numerical(student_answers, bundle):
    num_cols, cat_cols = bundle["num_cols"], bundle["cat_cols"]
    row = pd.DataFrame([student_answers])
    num_part = row[num_cols].astype(float).values
    cat_part = bundle["cat_encoder"].transform(row[cat_cols])
    X = np.hstack([num_part, cat_part])
    probs = bundle["model"].predict_proba(X)
    return probs[0], X


def _face_crop(img, mtcnn):
    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    eval_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        face_tensor = mtcnn(img)
    if face_tensor is not None:
        face_pil = to_pil((face_tensor + 1) / 2)
    else:
        min_dim = min(img.size)
        face_pil = T.Compose([T.CenterCrop(min_dim), T.Resize((IMAGE_SIZE, IMAGE_SIZE))])(img)
    return face_pil, eval_transform(face_pil).unsqueeze(0)


def predict_image(pil_image, model, mtcnn):
    face_pil, x = _face_crop(pil_image, mtcnn)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).numpy()[0]
    class_names = getattr(model, "_class_names", None)
    dep_idx = None
    if class_names:
        dep_idx = next((i for i, c in enumerate(class_names)
                         if "depress" in c.lower() and "not" not in c.lower()), None)
    p_dep = probs[dep_idx] if dep_idx is not None else (probs[1] if len(probs) > 1 else probs[0])
    return p_dep, face_pil


K_SEVERITY_SLOPE = 0.6
M_NORMAL_BOOST = 0.6


def normalize(v, eps=1e-9):
    v = np.clip(v, eps, None)
    return v / v.sum()


def build_image_prior(p_dep):
    if p_dep is None:
        return np.ones(5) / 5
    prior_severity = np.array([1.0, 1 + K_SEVERITY_SLOPE, 1 + 2 * K_SEVERITY_SLOPE,
                                1 + 3 * K_SEVERITY_SLOPE, 1 + 4 * K_SEVERITY_SLOPE])
    prior_normal = np.array([1 + M_NORMAL_BOOST, 1.0, 1.0, 1.0, 1.0])
    log_prior = p_dep * np.log(prior_severity) + (1 - p_dep) * np.log(prior_normal)
    return normalize(np.exp(log_prior))


def fuse_one(text_probs, num_probs, image_p_dep, alpha=0.25, w_text=0.5, w_num=0.5, eps=1e-9):
    t = normalize(text_probs) if text_probs is not None else None
    m = normalize(num_probs) if num_probs is not None else None
    if t is None and m is None:
        combined = np.ones(5) / 5
    elif t is None:
        combined = m
    elif m is None:
        combined = t
    else:
        wsum = w_text + w_num
        combined = normalize((w_text / wsum) * t + (w_num / wsum) * m)
    image_prior = build_image_prior(image_p_dep)
    fused = normalize(np.power(combined + eps, 1 - alpha) * np.power(image_prior + eps, alpha))
    return fused, combined


# --------------------------------------------------------------------------- #
# 3. XAI (SHAP / Captum / Grad-CAM) -- return matplotlib figures for st.pyplot
# --------------------------------------------------------------------------- #
def explain_numerical_fig(X, num_bundle, predicted_class_idx, top_k=10):
    import shap
    feature_names = num_bundle["num_cols"] + num_bundle["cat_cols"]
    explainer = shap.TreeExplainer(num_bundle["model"])
    raw_shap = explainer.shap_values(X)
    if isinstance(raw_shap, list):
        class_shap = raw_shap[predicted_class_idx][0]
    elif raw_shap.ndim == 3:
        class_shap = raw_shap[0, :, predicted_class_idx]
    else:
        class_shap = raw_shap[0]

    order = np.argsort(-np.abs(class_shap))[:top_k]
    top_features = [feature_names[i] for i in order]
    top_values = class_shap[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#E57373" if v > 0 else "#64B5F6" for v in top_values]
    ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"SHAP value (impact on '{LABELS[predicted_class_idx]}')")
    ax.set_title("Numerical model — top contributing features", fontweight="bold")
    plt.tight_layout()
    return fig


def explain_text_html(mental_state_text, future_thoughts_text, tokenizer, text_model,
                       predicted_class_idx, max_len=128):
    from captum.attr import LayerIntegratedGradients

    text = f"{mental_state_text.strip()} [SEP] {future_thoughts_text.strip()}"
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    pad_id = tokenizer.pad_token_id or 0
    baseline_ids = torch.full_like(input_ids, pad_id)
    baseline_ids[0, 0] = input_ids[0, 0]
    baseline_ids[0, -1] = input_ids[0, -1]

    def forward_func(ids, mask):
        return text_model(input_ids=ids, attention_mask=mask).logits

    lig = LayerIntegratedGradients(forward_func, text_model.get_input_embeddings())
    attributions, _ = lig.attribute(
        inputs=input_ids, baselines=baseline_ids, additional_forward_args=(attention_mask,),
        target=predicted_class_idx, n_steps=30, return_convergence_delta=True,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0)
    token_scores = (token_scores / (token_scores.norm() + 1e-9)).detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    real_len = int(attention_mask[0].sum().item())
    tokens, token_scores = tokens[:real_len], token_scores[:real_len]

    max_abs = np.max(np.abs(token_scores)) + 1e-9
    spans = []
    for tok, score in zip(tokens, token_scores):
        alpha = min(abs(float(score)) / max_abs, 1.0)
        color = f"rgba(229,115,115,{alpha:.2f})" if score > 0 else f"rgba(100,181,246,{alpha:.2f})"
        spans.append(f"<span style='background-color:{color};padding:2px 1px;border-radius:3px'>{tok}</span>")
    html = "<div style='line-height:2.2;font-size:16px'>" + " ".join(spans) + "</div>"

    top_idx = np.argsort(-np.abs(token_scores))[:10]
    top_rows = [(tokens[i], float(token_scores[i])) for i in sorted(top_idx)]
    return html, top_rows


def explain_image_fig(pil_image, img_model, mtcnn, predicted_class_idx=None):
    import torchvision.transforms as T
    face_pil, x = _face_crop(pil_image, mtcnn)

    class_names = getattr(img_model, "_class_names", None)
    if predicted_class_idx is None:
        predicted_class_idx = 1
        if class_names:
            predicted_class_idx = next(
                (i for i, c in enumerate(class_names) if "depress" in c.lower() and "not" not in c.lower()), 1)

    activations, gradients = {}, {}
    target_layer = img_model.layer4[-1]

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    img_model.zero_grad()
    logits = img_model(x)
    logits[0, predicted_class_idx].backward()
    h1.remove()
    h2.remove()

    acts = activations["value"][0]
    grads = gradients["value"][0]
    weights = grads.mean(dim=(1, 2))
    camv = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    camv = (camv - camv.min()) / (camv.max() - camv.min() + 1e-9)
    camv = camv.detach().numpy()
    cam_img = Image.fromarray(np.uint8(255 * camv)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    face_np = np.array(face_pil.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam_resized)[..., :3]
    overlay = 0.55 * face_np + 0.45 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(face_np); axes[0].set_title("Face-cropped input"); axes[0].axis("off")
    axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("Grad-CAM heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. AUTO-SAVE (every student's input + fused output -> CSV, appended)
# --------------------------------------------------------------------------- #
def entropy(probs, eps=1e-9):
    p = np.clip(probs, eps, None)
    return float(-np.sum(p * np.log2(p)))


def save_prediction_log(student_id, mental_state_text, future_thoughts_text, numerical_features,
                         has_image, text_probs, num_probs, p_dep, combined, fused_probs, final_label):
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "student_id": student_id,
        "mental_state_text": mental_state_text,
        "future_thoughts_text": future_thoughts_text,
        "has_image": has_image,
    }
    if numerical_features:
        row.update(numerical_features)
    for i, lbl in enumerate(LABELS):
        row[f"text_prob_{lbl}"] = round(float(text_probs[i]), 4) if text_probs is not None else None
    for i, lbl in enumerate(LABELS):
        row[f"num_prob_{lbl}"] = round(float(num_probs[i]), 4) if num_probs is not None else None
    row["image_p_depression"] = round(float(p_dep), 4) if p_dep is not None else None
    for i, lbl in enumerate(LABELS):
        row[f"fused_prob_{lbl}"] = round(float(fused_probs[i]), 4)
    row["final_predicted_label"] = final_label
    row["confidence"] = round(float(np.max(fused_probs)), 4)
    row["entropy_bits"] = round(entropy(fused_probs), 4)

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(LOG_PATH)
    df_row.to_csv(LOG_PATH, mode="a", header=write_header, index=False)

    # Optional: also mirror to a Google Sheet for TRUE persistence across
    # redeploys (Streamlit Cloud's local disk is wiped on restart/redeploy).
    # Configure st.secrets["gcp_service_account"] + st.secrets["GSHEET_URL"]
    # to enable this. See README.md.
    try:
        if "gcp_service_account" in st.secrets and "GSHEET_URL" in st.secrets:
            import gspread
            from google.oauth2.service_account import Credentials
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
            gc = gspread.authorize(creds)
            sh = gc.open_by_url(st.secrets["GSHEET_URL"]).sheet1
            if sh.row_count <= 1 and not sh.get_all_values():
                sh.append_row(list(row.keys()))
            sh.append_row([str(v) for v in row.values()])
    except Exception as e:  # noqa: BLE001
        st.info(f"(Google Sheet sync skipped: {e})")

    return row


# --------------------------------------------------------------------------- #
# 5. UI
# --------------------------------------------------------------------------- #
st.title("🎓 Student Mental-Health Screener (Text + Numerical + Image Fusion)")
st.caption("For research / demo purposes only. Not a diagnostic tool.")

tab_predict, tab_history, tab_about = st.tabs(["🧪 New Prediction", "📜 Saved History", "ℹ️ About"])

with tab_predict:
    with st.form("student_form"):
        st.subheader("1. Student info")
        student_id = st.text_input("Student ID / Roll (optional)", "")

        st.subheader("2. Text (Bangla/English, mixed OK)")
        mental_state_text = st.text_area("Present mental state (কেমন লাগছে আজকাল)", height=90)
        future_thoughts_text = st.text_area("Thoughts about the future (ভবিষ্যৎ নিয়ে চিন্তা)", height=90)

        st.subheader("3. Numerical / lifestyle answers")
        c1, c2, c3, c4 = st.columns(4)
        numerical_features = {}
        likert_cols = ["Academic_Stress", "Financial_Stress", "Anxiety_Level", "Loneliness_Level",
                        "Family_Support", "Self_Confidence", "Appearance_Satisfaction",
                        "Academic_Frustration", "Social_Media_Comparison", "Course_Retake_Stress",
                        "Career_Anxiety", "Living_Env_Satisfaction"]
        cols_cycle = [c1, c2, c3, c4]
        for i, col in enumerate(likert_cols):
            numerical_features[col] = cols_cycle[i % 4].slider(col.replace("_", " "), 1, 5, 3)

        numerical_features["Age"] = c1.number_input("Age", 16, 40, 21)
        numerical_features["CGPA"] = c2.number_input("CGPA", 0.0, 4.0, 3.2, step=0.01)
        numerical_features["Sleep_Hours"] = c3.number_input("Sleep hours/night", 0.0, 14.0, 6.0, step=0.5)
        numerical_features["Screen_Time"] = c4.number_input("Screen time (hrs/day)", 0.0, 20.0, 6.0, step=0.5)

        c5, c6 = st.columns(2)
        with c5:
            for col in CAT_COLS[:5]:
                numerical_features[col] = st.selectbox(col.replace("_", " "), CAT_OPTIONS[col], key=f"cat_{col}")
        with c6:
            for col in CAT_COLS[5:]:
                numerical_features[col] = st.selectbox(col.replace("_", " "), CAT_OPTIONS[col], key=f"cat_{col}")

        st.subheader("4. Face photo (optional, for image modality)")
        uploaded_image = st.file_uploader("Upload a clear front-facing photo", type=["jpg", "jpeg", "png"])

        st.subheader("5. Fusion weights")
        wcol1, wcol2, wcol3 = st.columns(3)
        w_text = wcol1.slider("Text weight", 0.0, 1.0, 0.5)
        w_num = wcol2.slider("Numerical weight", 0.0, 1.0, 0.5)
        alpha = wcol3.slider("Image prior weight (alpha)", 0.0, 1.0, 0.25)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        if not mental_state_text.strip() or not future_thoughts_text.strip():
            st.error("দুইটা text field-ই পূরণ করুন।")
            st.stop()

        with st.spinner("Loading models & predicting..."):
            tokenizer, text_model = load_text_model()
            text_probs, text_inputs = predict_text(mental_state_text, future_thoughts_text, tokenizer, text_model)

            num_bundle = load_numerical_model()
            num_probs, X_num = predict_numerical(numerical_features, num_bundle)

            p_dep, face_pil, img_model, mtcnn = None, None, None, None
            if uploaded_image is not None:
                pil_image = Image.open(uploaded_image).convert("RGB")
                img_model = load_image_model()
                mtcnn = get_mtcnn()
                p_dep, face_pil = predict_image(pil_image, img_model, mtcnn)

            fused_probs, combined = fuse_one(text_probs, num_probs, p_dep, alpha=alpha, w_text=w_text, w_num=w_num)
            final_label = LABELS[int(np.argmax(fused_probs))]
            predicted_class_idx = int(np.argmax(fused_probs))

        st.success(f"### Final Predicted Severity: **{final_label}**")

        # ---- results chart ----
        colA, colB = st.columns([1, 2]) if face_pil is not None else (st.container(), st.container())
        if face_pil is not None:
            with colA:
                st.image(face_pil, caption="Face-cropped input", width=220)
            chart_col = colB
        else:
            chart_col = colB if face_pil is not None else st

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(LABELS))
        width = 0.25
        ax.bar(x - width, text_probs, width, label="Text-only", color="#64B5F6")
        ax.bar(x, num_probs, width, label="Numerical-only", color="#FFB74D")
        ax.bar(x + width, fused_probs, width, label="Final Fused", color="#E57373")
        ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=15)
        ax.set_ylabel("Probability"); ax.set_ylim(0, 1)
        ax.set_title("Predicted Severity Distribution")
        ax.legend()
        for i, v in enumerate(fused_probs):
            ax.text(x[i] + width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        (chart_col if face_pil is not None else st).pyplot(fig)

        # ---- auto-save ----
        row = save_prediction_log(student_id, mental_state_text, future_thoughts_text, numerical_features,
                                   uploaded_image is not None, text_probs, num_probs, p_dep, combined,
                                   fused_probs, final_label)
        st.caption(f"✅ Auto-saved to log (confidence={row['confidence']}, entropy={row['entropy_bits']} bits)")

        # ---- XAI ----
        st.markdown("---")
        st.subheader("🔍 Explainable AI — why this prediction?")

        with st.expander("Numerical model — SHAP feature importance", expanded=True):
            try:
                shap_fig = explain_numerical_fig(X_num, num_bundle, predicted_class_idx)
                st.pyplot(shap_fig)
            except Exception as e:  # noqa: BLE001
                st.warning(f"SHAP explanation unavailable: {e}")

        with st.expander("Text model — Captum token attributions", expanded=True):
            try:
                html, top_rows = explain_text_html(mental_state_text, future_thoughts_text,
                                                    tokenizer, text_model, predicted_class_idx)
                st.markdown(html, unsafe_allow_html=True)
                st.write("Top contributing tokens:")
                st.dataframe(pd.DataFrame(top_rows, columns=["token", "score"]))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Text explanation unavailable: {e}")

        if uploaded_image is not None:
            with st.expander("Image model — Grad-CAM", expanded=True):
                try:
                    cam_fig = explain_image_fig(pil_image, img_model, mtcnn, predicted_class_idx)
                    st.pyplot(cam_fig)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Grad-CAM explanation unavailable: {e}")

with tab_history:
    st.subheader("Saved student records (auto-saved)")
    if os.path.exists(LOG_PATH):
        hist_df = pd.read_csv(LOG_PATH)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("⬇️ Download full log (CSV)", hist_df.to_csv(index=False).encode("utf-8"),
                            file_name="predictions_log.csv", mime="text/csv")
    else:
        st.info("এখনো কোনো prediction save হয়নি।")

with tab_about:
    st.markdown("""
    **Pipeline**: XLM-RoBERTa (text) + XGBoost (numerical) + ResNet50 (face image),
    combined with a late-fusion rule, explained with SHAP / Captum Integrated
    Gradients / Grad-CAM.

    This is a screening/demo tool for research and coursework purposes — **not**
    a clinical diagnostic instrument. If you or someone you know is struggling,
    please reach out to a counselor or a mental-health professional.
    """)
