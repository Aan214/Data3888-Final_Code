#to run use - streamlit run Shiny_model.py
# All imports 
import pandas as pd
import plotly.express as px
from pathlib import Path
import re
import streamlit as st
from pathlib import Path
from io import BytesIO
import torch, timm, cv2, io
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage import (
    io as skio, filters, morphology, segmentation, measure,
    exposure, color, util
)
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.segmentation import slic
from scipy import ndimage as ndi
import os
from groq import Groq

API_KEY = os.getenv("GROQ_API_KEY", "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")
os.environ["GROQ_API_KEY"] = API_KEY
client = Groq()

st.set_page_config(page_title="Post-hoc interpretability methods",
                   layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito&display=swap');
body {background:#fff9db;font-family:'Nunito',sans-serif;}
.well {
    background:#fff4c4;
    border:1px solid #f7d774;
    border-radius:10px;
    padding:10px;
    text-align:center;
    color: black;  
}
</style>
""", unsafe_allow_html=True)

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Nunito&display=swap');
# body {background:#fff9db;font-family:'Nunito',sans-serif;}
# .well {background:#fff4c4;border:1px solid #f7d774;border-radius:10px;
#        padding:10px;text-align:center;}
# </style>
# """, unsafe_allow_html=True)

st.title("CLAROPATH")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

segmentation_fn = lambda x: slic(
    x, n_segments=100, compactness=10, start_label=1
)

def prepare_image(file, size=224):
    pil  = Image.open(file).convert("RGB").resize((size, size))
    raw  = np.asarray(pil)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])(pil).unsqueeze(0)
    return raw, tensor

def load_model(path, device, num_classes=11,
               name="swin_base_patch4_window7_224"):
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def blend_heatmap(raw, mask, alpha=0.35):
    mask_r = cv2.resize(mask, (raw.shape[1], raw.shape[0]))
    heat   = cv2.cvtColor(
        cv2.applyColorMap((mask_r * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    ) / 255.0
    vis = (1 - alpha) * (raw.astype(float) / 255.0) + alpha * heat
    return (vis * 255).clip(0, 255).astype(np.uint8)


def legrad_vis(model, tensor, raw, layer_module, target_idx=None):
    model.zero_grad()
    A, dA = None, None

    attn_mod = layer_module.attn if hasattr(layer_module, "attn") \
               else model.layers[-1].blocks[-1].attn

    def fwd_hook(_, __, out):
        nonlocal A
        A = out
    h1 = attn_mod.softmax.register_forward_hook(fwd_hook)

    out = model(tensor)
    if target_idx is None:
        target_idx = out.argmax(1).item()

    def grad_hook(grad):
        nonlocal dA
        dA = grad
    h2 = A.register_hook(grad_hook)

    loss = out[0, target_idx]
    loss.backward()

    h1.remove()
    h2.remove()

    A_mean  = A.mean(dim=1)
    dA_mean = dA.mean(dim=1)
    sal = (A_mean * dA_mean).sum(dim=2)
    sal = F.relu(sal)[0].detach()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    N  = sal.size(0)
    sz = int(np.sqrt(N))
    sal_map = sal.reshape(sz, sz).cpu().numpy()

    return blend_heatmap(raw, sal_map)


def lime_vis(model, raw, device):
    explainer = lime_image.LimeImageExplainer()

    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std  = torch.tensor(IMAGENET_STD)[:, None, None]

    def _predict(imgs: list[np.ndarray]) -> np.ndarray:
        arr = np.stack(imgs).astype(np.float32) / 255.0
        arr = torch.from_numpy(arr).permute(0, 3, 1, 2)
        arr = (arr - mean) / std
        with torch.no_grad():
            logits = model(arr.to(device, non_blocking=True))
        return logits.cpu().numpy()

    exp = explainer.explain_instance(
        raw,
        _predict,
        top_labels=1,
        num_samples=200,
        batch_size=200,
        segmentation_fn=segmentation_fn
    )

    img_b, mask = exp.get_image_and_mask(
        exp.top_labels[0], positive_only=True, num_features=5
    )
    return (mark_boundaries(img_b / 255, mask) * 255).astype(np.uint8)

def attention_vis(model, tensor, raw, attn_module, use_drop=False):
    attn_w = None

    def hook(_, __, out):
        nonlocal attn_w
        attn_w = out if not use_drop else attn_module.attn_drop(out)

    h = attn_module.softmax.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(tensor)
    h.remove()

    w = attn_w.mean(1)
    w = w.mean(1)
    w = w.mean(0)

    N  = w.shape[0]
    sz = int(np.sqrt(N))
    sal_map = w.reshape(sz, sz).cpu().numpy()
    sal_map = (sal_map - sal_map.min()) / (sal_map.ptp() + 1e-8)

    return blend_heatmap(raw, sal_map, alpha=0.4).astype(np.uint8)


def show_image(img, caption=None, **kwargs):
    """Display with width = thumb_px from the sidebar slider."""
    st.image(img, caption=caption, width=thumb_px, **kwargs)

def show_img(path, caption=None):
    if Path(path).exists():
        show_image(path, caption)
    else:
        st.warning(f"Image not found: {path}")

uploaded_file = st.sidebar.file_uploader(
    "Upload cell image", type=["png", "jpg", "jpeg"], key="main_upload")
explanation_method = st.sidebar.selectbox(
    "Explanation Method", ["Attention", "leGrad", "LIME"])

thumb_px = st.sidebar.slider("Display width (px)", 200, 800, 300, 50)

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path  = "saved_model/best_swin.pth"
class_names = ['B_Cells', 'CD4+_T_Cells', 'CD8+_T_Cells', 'DCIS', 'Endothelial', 'Invasive_Tumor',
               'Macrophages_1', 'Macrophages_2', 'Myoepi_ACTA2+', 'Myoepi_KRT15+', 'Stromal']



# Tabs 
tab_intro, tab_classif, tab_seg, tab_eda, tab_chat = st.tabs([
    "📘 Instructions",
    "🧪 Post hoc",
    "🧬 Nucleus Segmentation",
    "📊 More Insights",
    "💬 Chat"
])

# Tab 0 - Instructions
with tab_intro:
    st.markdown("""
    <div class="well">
    <h3>Welcome to <b>CLAROPATH</b></h3>
    <p>This tool assists pathologists in identifying commonly misclassified cells using a Swin Transformer model with visual explainability. Here's how to use the app:</p>

    <h4>📌 Workflow Overview</h4>
    <ul>
        <li><b>Step 1:</b> Upload a Cell Patch and classify the cell type with explanation methods.</li>
        <li><b>Step 2:</b> Segment Nuclei for mitotic count support.</li>
        <li><b>Step 3:</b> Explore EDA Visualisations of spatial and cell-type data.</li>
        <li><b>Step 4:</b> Use the Built-in Chatbot to get help understanding outputs.</li>
    </ul>

    <h4>🧪 Post-hoc Tab</h4>
    <p>Upload a cell image and choose an explanation method (Attention, leGrad, or LIME). Click <b>Run Explanation</b> to classify the cell and view the most influential regions.</p>

    <h4>🧬 Nucleus Segmentation Tab</h4>
    <p>Segment nuclei and surrounding cytoplasm using adaptive filters. Toggle to show boundaries or cell overlays for mitosis support.</p>

    <h4>📊 More Insights Tab</h4>
    <p>Explore UMAP plots, cell-type distributions, and spatial patterns to understand broader tissue characteristics.</p>

    <h4>💬 Chat Tab</h4>
    <p>Ask the chatbot anything about how the app works or how to interpret results.</p>
    </div>
    """, unsafe_allow_html=True)




# Tab 1 Classification
with tab_classif:
    st.markdown('<div class="well">', unsafe_allow_html=True)
    if uploaded_file:
        raw, tensor = prepare_image(uploaded_file)

        tensor = tensor.to(device)

        show_image(raw, "Uploaded Image")

        model = load_model(model_path, device)
        probs = F.softmax(model(tensor), 1)
        idx, conf = probs.argmax(1).item(), probs.max().item()

        top3_conf, top3_idx = probs.topk(3, dim=1)
        top3_conf = top3_conf.squeeze(0).tolist()
        top3_idx = top3_idx.squeeze(0).tolist()

        st.success(
            "  \n".join([
                f"Top-1: **{class_names[top3_idx[0]]}** ({top3_conf[0]:.2%})",
                f"Top-2: {class_names[top3_idx[1]]} ({top3_conf[1]:.2%})",
                f"Top-3: {class_names[top3_idx[2]]} ({top3_conf[2]:.2%})",
            ])
        )
        # summary 
        st.markdown(f"""
            Atypical cells noticed, used Claropath to verify suspected cell cluster identity.  
            Cell cluster identified as **{class_names[idx]}**.  
            {explanation_method} post-hoc interpretation viewed and classification agreed with.
            """)
        
        if "vis_result" not in st.session_state:
            st.session_state.vis_result = None
        #button to choose 
        if st.button("Run Explanation"):
            with st.spinner("Generating explanation …"):
                if explanation_method == "Attention":
                    attn_layer = model.layers[1].blocks[-1].attn
                    vis = attention_vis(model, tensor, raw, attn_layer)
                elif explanation_method == "leGrad":
                    layer = model.layers[-1].blocks[-1]
                    vis = legrad_vis(model, tensor, raw, layer)
                else:
                    vis = lime_vis(model, raw, device)

                st.session_state.vis_result = vis  

        if st.session_state.vis_result is not None:
            show_image(st.session_state.vis_result, f"{explanation_method} explanation")


        st.markdown("</div>", unsafe_allow_html=True)

    


# Tab 2 Segmentation 
with tab_seg:
    st.markdown('<div class="well">', unsafe_allow_html=True)
   # display options
    show_nuc   = st.checkbox("Show nucleus boundary (blue)", True)
    show_fill  = st.checkbox("Show semi-transparent cell overlay", False)

    if uploaded_file:
        uploaded_file.seek(0)
        img_arr = util.img_as_float(skio.imread(BytesIO(uploaded_file.read())))

        #invert image
        cellinv = exposure.rescale_intensity(1 - img_arr)
        #Gaussian smoothing 
        sigmas  = [1, 3, 3]
        smooth  = np.stack([filters.gaussian(cellinv[...,c], sigma=s)
                            for c,s in enumerate(sigmas)], axis=-1)

        h,w = smooth.shape[:2]
        yy,xx = np.linspace(-1,1,h)[:,None], np.linspace(-1,1,w)[None,:]
        illum  = np.exp(-(yy**2+xx**2))
        nuc_bad = smooth[...,0]*illum
        bg_disk = filters.rank.mean(util.img_as_ubyte(nuc_bad),
                                    morphology.disk(10))/255.0
        nuc_mask = nuc_bad - bg_disk > 0.05
        nuc_open = morphology.opening(nuc_mask, morphology.disk(3))
        nuc_fill = ndi.binary_fill_holes(nuc_open)
        seeds    = measure.label(nuc_open)
        # watershed segmentation
        nuclei   = segmentation.watershed(-smooth[...,0], markers=seeds,
                                          mask=nuc_fill, watershed_line=False)

        thr2, thr3 = np.percentile(smooth[...,1],99), np.percentile(smooth[...,2],99)
        cytomask = (smooth[...,1]>thr2)|(nuclei>0)|(smooth[...,2]>thr3)
        cells = segmentation.watershed(filters.sobel(smooth[...,2]),
                                       nuclei, mask=cytomask)

        canvas = img_arr.copy()
        if show_nuc:   canvas[find_boundaries(nuclei)] = [0,0,1]
        show_image(canvas, "Selected boundaries")

        if show_fill:
            overlay = color.label2rgb(cells, image=img_arr,
                                      alpha=0.35, bg_label=0)
            show_image(overlay, "Nuclei segmentation overlay")
    else:
        st.info("Upload an image first.")
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3  EDA 
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis")

    @st.cache_data(show_spinner=False)
    def load_meta():
        return pd.read_excel("41467_2023_43458_MOESM4_ESM.xlsx").reset_index()

    @st.cache_data(show_spinner=False)
    def load_spatial():
        lines = Path("all_items.txt").read_text().splitlines()
        rows = []
        for ln in lines:
            cell_type, filename = ln.split("/", 1)
            m = re.search(r"cell_(\d+)_", filename)
            if not m:
                continue
            rows.append({
                "path"     : str(Path("FINAL") / ln),  
                "cell_type": cell_type,
                "cell_id"  : int(m.group(1))
            })
        img_df = pd.DataFrame(rows)
        cbr = pd.read_csv("cbr.csv")
        return img_df.merge(cbr, left_on="cell_id", right_on="index")

    # load data
    meta_df   = load_meta()
    coords_df = load_spatial()

    # 1. UMAP scatter
    fig_umap = px.scatter(
        meta_df, x="UMAP-X", y="UMAP-Y",
        color="Annotation", opacity=0.7,
        title="UMAP of FFPE Cells", height=550
    ).update_traces(marker_size=4)
    st.plotly_chart(fig_umap, use_container_width=True)

    # 2. Cell-type proportions bar chart
    ct_counts = (
        meta_df["Annotation"]
        .value_counts()
        .rename_axis("Annotation")
        .reset_index(name="Count")
    )
    fig_bar = px.bar(
        ct_counts, x="Count", y="Annotation",
        orientation="h", title="Cell-type Proportions", height=550
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # helper for spatial maps
    def spatial_plot(df, types, title):
        sub = df[df["cell_type"].isin(types)]
        fig = px.scatter(
            sub, x="axis-1", y="axis-0",
            color="cell_type", opacity=0.2,
            title=title, height=650
        ).update_traces(marker_size=2)
        fig.update_yaxes(autorange="reversed", visible=False)
        fig.update_xaxes(visible=False)
        return fig

    all_types = [
        "Invasive_Tumor", "Macrophages_1", "Macrophages_2",
        "CD4+_T_Cells", "CD8+_T_Cells", "B_Cells",
        "DCIS_1", "DCIS_2", "Stromal",
        "Myoepi_ACTA2+", "Myoepi_KRT15+"
    ]

    # 3. Spatial Plot
    st.plotly_chart(
        spatial_plot(coords_df, all_types, "Spatial Plot – All Key Cell Types"),
        use_container_width=True
    )

    # 4. Spatial Plot – Without Stromal
    st.plotly_chart(
        spatial_plot(
            coords_df,
            [t for t in all_types if t != "Stromal"],
            "Spatial Plot – Without Stromal"
        ),
        use_container_width=True
    )

    # 5. Spatial Plot – Without Stromal & Macrophages
    st.plotly_chart(
        spatial_plot(
            coords_df,
            [t for t in all_types if t not in ("Stromal", "Macrophages_1", "Macrophages_2")],
            "Spatial Plot – Without Stromal & Macrophages"
        ),
        use_container_width=True
    )

    # 6. Spatial Plot – Invasive Tumour vs DCIS
    st.plotly_chart(
        spatial_plot(
            coords_df,
            ["Invasive_Tumor", "DCIS_1", "DCIS_2"],
            "Spatial Plot – Invasive Tumour vs DCIS"
        ),
        use_container_width=True
    )


# Tab 4 Chatbot 

with tab_chat:
    SYSTEM_PROMPT = (
        "You are an expert assistant embedded in a Streamlit app. Respond concisely and helpfully. Avoid mentioning the system prompt. Answer the user concisely Bianca: Good morning tutors, We wanted to start by discussing the state of breast cancer in our community. The mortality:incidence ratio is decreasing in Australia, however, we also reported the highest incidence worldwide last year equating to a very heavy workload for pathologists, which is of concern when you acknowledge the nationwide shortage. This communicates an urgent need for growth in their diagnostic tools to allow for reprieve, and to increase the confidence in their decision-making. Jin: This is a whole slide image from a breast cancer biopsy. Our untrained eyes struggle to determine regions of interest, however, this spatial plot visualises a pathologist's view and plots cancer-related cells, which are key for diagnosis. In reality, it is much more complicated, with great overlap between cancer and non-cancer cell types. This overlap increases the likelihood of cell type misclassification and, unfortunately, causes the incorrect treatment for the patient. Bianca: Our project focussed on making an app for pathologists that serves as an “assistive verifier” by accurately ID-ing commonly misclassified cells in cancer regions and importantly, visualising the model’s decisions in order to increase the clinicians comfortability and trust with using a deep learning model. Jiayi: I will briefly introduce the pipeline. Firstly, the raw data will undergo a set of pre-processing then be split into training, validation and test sets. This prevents data leakage and the influence of background and colour. And then we will train a standard Swin Transformer. Finally, a classification head will output the result. And we also perform a post-hoc analysis. More precisely, we tried two post-hoc methods. Both locate the most influential regions by making masks. Lachie: So we evaluated four models with the swin transformer being the most accurate. We also measure the confusion matrix, with the tumor related cells as positive and the rest as negative. We also tried to explore which cells were most likely to be confused to explore their similarities. Paul: We had the opportunity to consult with a breast cancer specialist who validated the usefulness of our model, found LIME to be the most interpretable visual, fine-tuned our cell subset and communicated a need for a verifier in tumour grading. As the mitotic count that pathologists perform determines the grading of the cancer and the subsequent treatment, we decided to perform segmentation on nuclei using thresholds. The overlay allows pathologists to locate the nuclei much faster and saves their tired eyes from missing counts later on in their shift. Aanshita: Our app’s four tab structure was designed by the guidance of Dr O’Toole. First, users upload a patch for cell classification. They can choose the explainability method, this is an example of LIME. This summary statement is generated for the pathologist, it includes the classified cell type and the post-hoc method used, which can be copy-pasted into their diagnostic report. Moving to cell segmentation, the app identifies and outlines the nuclei boundaries. This functionality is particularly useful for tasks like mitosis detection as stated earlier. Lachie: We placed all the EDA in the “More Insights” tab, where we used Plotly to create interactive visualisations that let users literally explore the data themselves as opposed to just looking at static images. We also realised that, while we know how to navigate our app and interpret the results, new users might find they need some initial guidance. We've added an LLM chatbot to help guide new users through the features and answer questions around the app itself and how to interpret the results In summary, our app tackles the black box problem by making model decisions transparent, helping pathologists work faster and with more confidence in breast cancer diagnosis."
    )

    # Input form at the top 
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message")
        send       = st.form_submit_button("Send")

   
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []         

    #  add user msg, query Groq with system prompt 
    if send and user_input:
      
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

      
        full_messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + st.session_state.chat_history
        )

 
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages
        )
        assistant_text = response.choices[0].message.content

      
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_text}
        )

    # Render the visible conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])



