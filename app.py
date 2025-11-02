import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
import numpy as np

# Configure the page
st.set_page_config(
    page_title="OilSpillDetect - Satellite Imagery Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .upgrade-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .threshold-slider {
        margin: 1rem 0;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .image-box {
        margin: 0.5rem;
        text-align: center;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    .detection-results {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Oil Spill Detection from Satellite Imagery</h1>', unsafe_allow_html=True)
st.markdown("**Advanced AI-powered detection with comprehensive metrics, loss functions, and data augmentation**")

# Sidebar for navigation and settings
with st.sidebar:
    st.header("Navigation & Settings")
    
    # Object selection
    st.subheader("Object")
    object_id = st.radio("ObjectID:", ["OilSpillDetect", "+ Upgrade"])
    
    # Extracted stack files status
    st.subheader("Extracted Stack Files")
    progress_value = 7/8
    st.progress(progress_value)
    st.write(f"7 / 8 files processed")
    st.warning("Handling the threshold change properly, causing a technical pitch on the server that prevents the results from updating dynamically.")
    
    # Restart options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restart 0.1 Oil Detector"):
            st.success("Oil detector restarted successfully!")
    with col2:
        if st.button("Restart Detection"):
            st.success("Detection process restarted!")
    
    # Status message
    st.error("Agent encountered an error while running, we are investigating the issue.")
    
    # Checkpoint and actions
    st.info("Checkpoint made 3 hours ago")
    st.write("Add comprehensive metrics, loss functions, and data augmentation visualization")
    
    # Rollback and preview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.checkbox("Rollback here")
    with col2:
        st.checkbox("Changes")
    with col3:
        st.checkbox("Preview")
    
    # Upgrade card
    st.markdown("---")
    st.markdown("""
    <div class="upgrade-card">
        <h3>Upgrade to continue building</h3>
        <p>You've reached your Skarter usage limit. Upgrade to make, launch, and scale your apps.</p>
        <ul>
            <li>$25 Monthly credits for Right Agent</li>
            <li>Publish and host your apps</li>
            <li>Access more powerful models</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Upgrade now ($10 off)"):
        st.success("Upgrade process initiated!")

# Main content area
tab1, tab2, tab3 = st.tabs(["Detection", "Analysis", "Training Metrics"])

with tab1:
    st.header("Upload & Settings")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Satellite Image", type=['png', 'jpg', 'jpeg', 'tiff'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Satellite Image", use_column_width=True)
    
    # Detection threshold
    st.subheader("Detection Threshold (%)")
    threshold = st.slider(
        "Adjust sensitivity for all oil spill detections (0–30%)", 
        min_value=0, 
        max_value=30, 
        value=15,
        key="threshold_slider"
    )
    
    # Detect button
    if st.button("Detect Oil Spill", type="primary"):
        if uploaded_file is not None:
            st.success("Oil spill detection in progress...")
            # Simulate API call
            # response = requests.post("YOUR_API_ENDPOINT", files={"image": uploaded_file})
        else:
            st.warning("Please upload a satellite image first.")
    
    # Detection Results
    st.markdown("---")
    st.header("Detection Results")
    
    # Create metrics in a grid
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">56.88%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">3.15%</div><div class="metric-label">IoU Score</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">6.11%</div><div class="metric-label">Dice Coefficient</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">6.11%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card"><div class="metric-value">32.37%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="metric-card"><div class="metric-value">3.37%</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
    with col7:
        st.markdown('<div class="metric-card"><div class="metric-value">94.98%</div><div class="metric-label">Specificity</div></div>', unsafe_allow_html=True)

with tab2:
    st.header("Image Analysis")
    
    # Image comparison section
    st.subheader("Visual Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original Image**")
        # Placeholder for original image
        st.image("https://via.placeholder.com/300x200/4A90E2/FFFFFF?text=Original+Image", use_column_width=True)
    
    with col2:
        st.markdown("**Ground Truth Mask**")
        # Placeholder for ground truth
        st.image("https://via.placeholder.com/300x200/7ED321/FFFFFF?text=Ground+Truth", use_column_width=True)
    
    with col3:
        st.markdown("**Prediction Overlay (Red)**")
        # Placeholder for prediction
        st.image("https://via.placeholder.com/300x200/D0021B/FFFFFF?text=Oil+Spill+Detection", use_column_width=True)
    
    # Oil spill area metrics
    st.subheader("Oil Spill Area Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">4.34%</div><div class="metric-label">Oil Spill Area</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">56.88%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">3.15%</div><div class="metric-label">IoU Score</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">6.11%</div><div class="metric-label">Dice Coefficient</div></div>', unsafe_allow_html=True)

with tab3:
    st.header("Training Metrics & Data Augmentation")
    
    # Data augmentation preview
    st.subheader("Data Augmentation Preview")
    st.write("Augmented versions of your uploaded image (changes dynamically based on input)")
    
    # Create sample augmentation examples
    aug_col1, aug_col2, aug_col3, aug_col4 = st.columns(4)
    
    with aug_col1:
        st.markdown("**Brightness Up**")
        st.image("https://via.placeholder.com/200x150/FF6B6B/FFFFFF?text=Brightness+Up", use_column_width=True)
        st.caption("0.00MB")
    
    with aug_col2:
        st.markdown("**Brightness Down**")
        st.image("https://via.placeholder.com/200x150/4ECDC4/FFFFFF?text=Brightness+Down", use_column_width=True)
        st.caption("0.00MB")
    
    with aug_col3:
        st.markdown("**Contrast Up**")
        st.image("https://via.placeholder.com/200x150/45B7D1/FFFFFF?text=Contrast+Up", use_column_width=True)
        st.caption("0.00MB")
    
    with aug_col4:
        st.markdown("**Rotation**")
        st.image("https://via.placeholder.com/200x150/96CEB4/FFFFFF?text=Rotation", use_column_width=True)
        st.caption("0.00MB")
    
    # Training graphs
    st.subheader("Training Progress Metrics")
    
    # Create sample data for graphs
    epochs = list(range(1, 101))
    
    # Dice Coefficient over epochs
    dice_data = [0.05 + 0.0006 * i + 0.0001 * (i % 10) for i in epochs]
    
    # Accuracy over epochs
    accuracy_data = [0.50 + 0.002 * i + 0.001 * (i % 15) for i in epochs]
    
    # IoU over epochs
    iou_data = [0.03 + 0.0003 * i + 0.0001 * (i % 20) for i in epochs]
    
    # Create columns for graphs
    graph_col1, graph_col2, graph_col3 = st.columns(3)
    
    with graph_col1:
        # Dice Coefficient graph
        fig_dice = go.Figure()
        fig_dice.add_trace(go.Scatter(x=epochs, y=dice_data, mode='lines', name='Dice Coefficient', line=dict(color='#FF6B6B')))
        fig_dice.update_layout(
            title='Dice Coefficient over Epochs',
            xaxis_title='Epochs',
            yaxis_title='Dice Coefficient',
            height=300
        )
        st.plotly_chart(fig_dice, use_container_width=True)
    
    with graph_col2:
        # Accuracy graph
        fig_accuracy = go.Figure()
        fig_accuracy.add_trace(go.Scatter(x=epochs, y=accuracy_data, mode='lines', name='Accuracy', line=dict(color='#4ECDC4')))
        fig_accuracy.update_layout(
            title='Accuracy over Epochs',
            xaxis_title='Epochs',
            yaxis_title='Accuracy',
            height=300
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with graph_col3:
        # IoU graph
        fig_iou = go.Figure()
        fig_iou.add_trace(go.Scatter(x=epochs, y=iou_data, mode='lines', name='IoU Score', line=dict(color='#45B7D1')))
        fig_iou.update_layout(
            title='IoU Score over Epochs',
            xaxis_title='Epochs',
            yaxis_title='IoU Score',
            height=300
        )
        st.plotly_chart(fig_iou, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### OilSpillDetect - Advanced AI-powered Oil Spill Detection System")