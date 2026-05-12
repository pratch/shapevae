import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import clip
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import trimesh
import tempfile
import os
from pathlib import Path

# Import your models
import sys
sys.path.insert(0, "/home/palakons/shapevae")
from model.base_model import PointCloudAE
from model.ptv3_based_model import AECLIPProjectionHead

# ==================== CONFIG ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 1024
NUM_POINTS = 1024
CLIP_DIM = 512

WEIGHTS_DIR = Path("/shapevae_weights")
CLIP2Z_ADAPTOR_PATH = WEIGHTS_DIR / "final_clip2z_adaptor_adaptor_clip2z_9y2o6vi2.pt" 
Z2CLIP_ADAPTOR_PATH = WEIGHTS_DIR / "final_adaptor_adaptor_z2clip_9y2o6vi2.pt"

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_clip_model():
    """Load pretrained CLIP model"""
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess

@st.cache_resource
def load_clip2z_adaptor():
    """Load CLIP embedding → z latent adaptor"""
    adaptor = AECLIPProjectionHead(input_dim=CLIP_DIM, output_dim=Z_DIM).to(DEVICE)
    adaptor.load_state_dict(torch.load(CLIP2Z_ADAPTOR_PATH, map_location=DEVICE))
    adaptor.eval()
    return adaptor

@st.cache_resource
def load_z2clip_adaptor():
    """Load z latent → CLIP embedding adaptor (for retrieval)"""
    adaptor = AECLIPProjectionHead(input_dim=Z_DIM, output_dim=CLIP_DIM).to(DEVICE)
    adaptor.load_state_dict(torch.load(Z2CLIP_ADAPTOR_PATH, map_location=DEVICE))
    adaptor.eval()
    return adaptor

@st.cache_resource
def load_point_ae(ae_checkpoint_path):
    """Load PointCloudAE model"""
    model = PointCloudAE(z_dim=Z_DIM, num_points=NUM_POINTS).to(DEVICE)
    checkpoint = torch.load(ae_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ==================== EMBEDDING & DECODING ====================
def text_to_z(text, clip_model, clip2z_adaptor, preprocess):
    """Convert text → CLIP embedding → z latent"""
    with torch.no_grad():
        text_tokens = clip.tokenize(text).to(DEVICE)
        text_embedding = clip_model.encode_text(text_tokens)  # [1, 512], float16
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.float()  # Convert to float32 for adaptor
        
        z = clip2z_adaptor(text_embedding)  # [1, Z_DIM]
    return z

def obj_to_z(obj_path, point_ae):
    """Load .obj file and convert to z latent"""
    mesh = trimesh.load(obj_path)
    
    # Sample points from mesh surface
    if isinstance(mesh, trimesh.Trimesh):
        points = mesh.sample(NUM_POINTS)
    else:
        # Handle point cloud files
        points = np.asarray(mesh.vertices)[:NUM_POINTS]
    
    # Normalize to [-1, 1]
    points = points - points.mean(axis=0)
    points = points / (np.linalg.norm(points, axis=0).max() + 1e-6)
    
    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        z = point_ae.encoder(points_tensor)  # [1, Z_DIM]
    
    return z, points

def z_to_pointcloud(z, point_ae):
    """Decode z latent to point cloud"""
    with torch.no_grad():
        pc = point_ae.decoder(z)  # [B, NUM_POINTS, 3]
    return pc.cpu().numpy()

# ==================== INTERPOLATION & EXTRAPOLATION ====================
def interpolate_embeddings(z_neg, z_pos, num_steps=5, extrapolate_factor=0.5):
    """
    Interpolate from negative to positive z embedding, with extrapolation.
    
    Args:
        z_neg: negative z embedding [1, Z_DIM]
        z_pos: positive z embedding [1, Z_DIM]
        num_steps: number of interpolation steps (middle)
        extrapolate_factor: how far to extrapolate beyond endpoints
    
    Returns:
        List of z embeddings: [extra_left, ..., z_neg, interp, ..., z_pos, ..., extra_right]
    """
    z_neg = z_neg.squeeze(0)  # [Z_DIM]
    z_pos = z_pos.squeeze(0)  # [Z_DIM]
    
    # Direction vector
    direction = z_pos - z_neg
    
    # Extrapolation to the left
    num_extra = num_steps // 2
    z_list = []
    for i in range(num_extra, 0, -1):
        t = -i * extrapolate_factor
        z_extra = z_neg + t * direction
        z_list.append(z_extra)
    
    # Interpolation from neg to pos
    for t in np.linspace(0, 1, num_steps):
        z_interp = z_neg + t * direction
        z_list.append(z_interp)
    
    # Extrapolation to the right
    for i in range(1, num_extra + 1):
        t = 1 + i * extrapolate_factor
        z_extra = z_neg + t * direction
        z_list.append(z_extra)
    
    # Convert to batch tensor
    z_batch = torch.stack(z_list).unsqueeze(1)  # [total_steps, 1, Z_DIM] → [total_steps, Z_DIM]
    return torch.cat([z.unsqueeze(0) for z in z_list], dim=0).to(DEVICE)

# ==================== 3D VISUALIZATION ====================
def create_pointcloud_trace(points, name, color="blue", size=2, opacity=0.8):
    """Create a Plotly scatter trace for a point cloud"""
    # points: [N, 3]
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        ),
        hoverinfo='skip',
    )
    return trace


def create_pointcloud_figure(points, title="", color="dodgerblue"):
    fig = go.Figure(data=[create_pointcloud_trace(points, title or "Shape", color=color)])
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=28 if title else 0, b=0),
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            aspectmode='data',
        ),
    )
    return fig

def create_synchronized_figure(pc_rows, titles, num_cols=None):
    """
    Create a grid of 3D point cloud visualizations with synchronized rotation.
    
    Args:
        pc_rows: List of lists of point clouds. Each inner list is a row.
        titles: Corresponding titles for each subplot
        num_cols: Number of columns per row
    """
    num_rows = len(pc_rows)
    if num_cols is None:
        num_cols = len(pc_rows[0]) if pc_rows else 1
    
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        specs=[[{'type': 'scatter3d'} for _ in range(num_cols)] for _ in range(num_rows)],
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    trace_idx = 0
    for row_idx, pc_row in enumerate(pc_rows):
        for col_idx, pc in enumerate(pc_row):
            trace = create_pointcloud_trace(pc, name=f"Shape {trace_idx}", color="dodgerblue")
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)
            
            # Set camera for each subplot (will be synchronized)
            fig.update_layout({f'scene{trace_idx + 1}': dict(
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                zaxis=dict(showgrid=False, zeroline=False),
                aspectmode='data',
            )})
            trace_idx += 1
    
    fig.update_layout(
        height=300 + num_rows * 400,
        width=1200,
        showlegend=False,
        hovermode='closest',
    )
    
    return fig


def render_synced_pointcloud_gallery(pointclouds, titles, height=760, n_cols=4):
    """Render separate point cloud plots with synced camera updates after drag settles."""
    if len(pointclouds) != len(titles):
        raise ValueError("pointclouds and titles must have the same length")

    cards = []
    for idx, (points, title) in enumerate(zip(pointclouds, titles)):
        fig = create_pointcloud_figure(points, title=title)
        fig.update_layout(height=300, width=300)
        cards.append(
            f'<div class="shape-card" data-shape-index="{idx}">'
            + pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn",
                config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                div_id=f"shape_plot_{idx}",
            )
            + "</div>"
        )

    # Build card wrappers and compute parameters OUTSIDE the loop
    card_wrappers = ''.join([
        f'<div class="shape-card-wrapper"><div class="shape-label">{title}</div>{card}</div>'
        for title, card in zip(titles, cards)
    ])
    
    # Build plot IDs array
    plot_ids = ', '.join([f'"shape_plot_{i}"' for i in range(len(pointclouds))])
    
    # Compute CSS flex percentage
    flex_pct = 100 / max(1, n_cols)
    
    # Lay out cards in rows
    num_items = len(pointclouds)
    rows = (num_items + max(1, n_cols) - 1) // max(1, n_cols)
    card_height = 340
    required_height = max(height, rows * card_height + 80)

    html = f"""
    <style>
        .path-container {{
            border: 1px solid rgba(255,255,255,0.12);
            padding: 0.5rem;
            border-radius: 6px;
            background: rgba(0,0,0,0.03);
            box-sizing: border-box;
        }}
        .shape-gallery {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: flex-start;
            justify-content: flex-start;
            padding-bottom: 0.25rem;
            width: 100%;
            box-sizing: border-box;
            overflow: visible;
        }}
        .shape-card-wrapper {{
            flex: 0 0 calc({flex_pct}% - 0.5rem);
            box-sizing: border-box;
        }}
        .shape-card {{
            width: 100%;
            min-width: 0;
        }}
        .shape-label {{
            font-size: 0.8rem;
            line-height: 1.1;
            margin: 0 0 0.15rem 0.1rem;
            color: rgba(255, 255, 255, 0.72);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .shape-card .js-plotly-plot {{
            width: 100% !important;
        }}
        @media (max-width: 768px) {{
            .shape-card-wrapper {{ flex: 0 0 calc(50% - 0.5rem); }}
        }}
        @media (max-width: 480px) {{
            .shape-card-wrapper {{ flex: 0 0 calc(100% - 0.5rem); }}
        }}
    </style>
    <div class="path-container">
      <div class="shape-gallery">
        {''.join([f'<div class="shape-card-wrapper"><div class="shape-label">{title}</div>{card}</div>' for title, card in zip(titles, cards)])}
      </div>
    </div>
    <script>
    (function () {{
        function waitForPlotly() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(waitForPlotly, 100);
                return;
            }}

            const plotIds = [{', '.join([f'"shape_plot_{i}"' for i in range(len(pointclouds))])}];
            const plots = plotIds.map((id) => document.getElementById(id)).filter(Boolean);
            if (!plots.length) return;

            const syncState = {{ timer: null }};

            function getCameraFromRelayout(gd, eventData) {{
                const sceneKeys = Object.keys(eventData || {{}}).filter((key) => /^(scene\d*|scene)\.camera\./.test(key));
                if (!sceneKeys.length) return null;
                const sourceScene = sceneKeys[0].split('.camera.')[0];
                const sourceLayout = gd._fullLayout[sourceScene];
                return sourceLayout && sourceLayout.camera ? sourceLayout.camera : null;
            }}

            function applyCameraToAll(camera, sourceId) {{
                plots.forEach((gd) => {{
                    if (!gd || gd.id === sourceId) return;
                    Plotly.relayout(gd, {{ 'scene.camera': camera }});
                }});
            }}

            plots.forEach((gd) => {{
                gd.on('plotly_relayout', function (eventData) {{
                    const camera = getCameraFromRelayout(gd, eventData);
                    if (!camera) return;

                    syncState.sourceId = gd.id;
                    window.clearTimeout(syncState.timer);
                    syncState.timer = window.setTimeout(() => {{
                        applyCameraToAll(camera, syncState.sourceId);
                    }}, 90);
                }});
            });
        }
        waitForPlotly();
    })();
    </script>
    """
    components.html(html, height=height, scrolling=True)

# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(page_title="Shape Interpolation", layout="wide", initial_sidebar_state="collapsed")
    
    # Maximize screen real estate with custom CSS
    st.markdown("""
    <style>
        .main { padding-top: 1rem; padding-bottom: 0; }
        [data-testid="stAppViewContainer"] { max-width: 100vw; padding: 0; }
        [data-testid="stMainBlockContainer"] { padding: 3.5rem 1rem; max-width: 100vw; }
        section[data-testid="stSidebar"] > div { padding-top: 0; }
    </style>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        clip_model, preprocess = load_clip_model()
        clip2z_adaptor = load_clip2z_adaptor()
        z2clip_adaptor = load_z2clip_adaptor()
    
    # Sidebar: Configuration and upload
    with st.sidebar.expander("Config", expanded=False):
        ae_checkpoint = st.text_input(
            "PointCloudAE checkpoint path",
            "/shapevae_weights/20260507-102556_palakons_baseline_cd_allcat_4interp_1000epoch/checkpoints/last.pt",
        )
        uploaded_file = st.file_uploader("Upload shape (.obj)", type=["obj", "ply"])
    
    try:
        point_ae = load_point_ae(ae_checkpoint)
    except Exception as e:
        st.error(f"Failed to load PointCloudAE: {e}")
        return
    
    # Single compact control row
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 0.8, 0.8, 0.6], gap="small")
    
    with col1:
        positive_text = st.text_input("Positive", value="a tall chair", label_visibility="collapsed")
    
    with col2:
        negative_text = st.text_input("Negative", value="a short chair", label_visibility="collapsed")
    
    with col3:
        num_interp_steps = st.slider("Steps", 3, 15, 7, label_visibility="collapsed")
    
    with col4:
        extrapolate_factor = st.slider("Extrap", 0.1, 1.0, 0.5, step=0.1, label_visibility="collapsed")
    
    with col5:
        generate = st.button("Generate", use_container_width=True)
    
    # Generate button
    if generate:
        with st.spinner("Processing..."):
            # Convert texts to z embeddings
            # st.info("Converting text prompts to embeddings...")
            z_positive = text_to_z(positive_text, clip_model, clip2z_adaptor, preprocess)
            z_negative = text_to_z(negative_text, clip_model, clip2z_adaptor, preprocess)
            
            # First row: Query shape
            # st.subheader("Query")
            
            if uploaded_file is not None:
                # Use uploaded shape
                with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                z_query, query_pc = obj_to_z(tmp_path, point_ae)
                query_pcs = [query_pc]
                query_titles = ["Uploaded Shape"]
                os.unlink(tmp_path)
            else:
                # Use closest to positive text
                z_query = z_positive
                query_pcs = [z_to_pointcloud(z_query, point_ae)[0]]
                query_titles = ["Closest to Positive Text"]
            
            # Display first row
            query_fig = create_pointcloud_figure(query_pcs[0], title="")
            query_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420, width=520)
            st.plotly_chart(query_fig, use_container_width=True, config={"displaylogo": False, "responsive": True})
            
            # Second row: Interpolations with extrapolation
            st.subheader("Path")
            
            z_interpolated = interpolate_embeddings(z_negative, z_positive, num_interp_steps, extrapolate_factor)
            interp_pcs = [z_to_pointcloud(z.unsqueeze(0), point_ae)[0] for z in z_interpolated]
            
            # Create titles for interpolation steps
            total_steps = len(interp_pcs)
            middle_idx = total_steps // 2
            interp_titles = []
            for i in range(total_steps):
                if i < middle_idx - num_interp_steps // 2:
                    interp_titles.append(f"Extra ← {i}")
                elif i > middle_idx + num_interp_steps // 2:
                    interp_titles.append(f"Extra → {i - middle_idx - num_interp_steps // 2}")
                else:
                    t = (i - middle_idx + num_interp_steps // 2) / (num_interp_steps - 1)
                    interp_titles.append(f"t={t:.2f}")
            
            # Render separate plots in a single scrollable strip so extrapolations stay visible.
            gallery_height = 380
            render_synced_pointcloud_gallery(interp_pcs, interp_titles, height=gallery_height, n_cols=len(interp_pcs))
            
            st.success("✅ Generation complete!")

if __name__ == "__main__":
    main()

