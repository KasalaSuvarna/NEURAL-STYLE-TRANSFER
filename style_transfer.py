import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Image preprocessing and postprocessing
def load_image(img, max_size=400):
    image = Image.open(img).convert('RGB')
    size = max(image.size)
    if size > max_size:
        image = image.resize((max_size, max_size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().squeeze()
    image = image.numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    return image

# Feature extraction and style loss
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Style transfer function
def style_transfer(content_img, style_img, steps=300):
    content = load_image(content_img)
    style = load_image(style_img)
    target = content.clone().requires_grad_(True)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1
    }
    content_weight = 1e4
    style_weight = 1e2

    optimizer = optim.Adam([target], lr=0.003)

    for i in range(steps):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_weights:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return tensor_to_image(target)
st.title("🎨 Neural Style Transfer")
st.write("Upload a content image and a style image to blend them.")

content_file = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'])
style_file = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'])

if content_file and style_file:
    st.image(content_file, caption="Content Image", width=300)
    st.image(style_file, caption="Style Image", width=300)

    if st.button("Stylize!"):
        with st.spinner("Stylizing... please wait..."):
            output_image = style_transfer(content_file, style_file)
            st.image(output_image, caption="Stylized Image", use_column_width=True)
            st.success("Done!")
