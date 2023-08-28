import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load the trained model
model = torch.load('model.pt')
model.load_state_dict(torch.load('modelSD.pt'))
model.eval()

# Set up image transformation
transform = transforms.Compose([
    transforms.Resize(32,32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

st.title("CIFAR-10 VAE Demo")

# Upload an image
uploaded_file = st.file_uploader("Choose a CIFAR-10 image to encode and decode", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file)
    image = transform(image).unsqueeze(0)

    # Encode and decode the image
    with torch.no_grad():
        _, reconstructed = model(image)

    # Convert the tensor to a PIL Image
    reconstructed = (reconstructed.squeeze(0) * 0.5) + 0.5
    reconstructed = transforms.ToPILImage()(reconstructed)

    # Display the reconstructed image
    st.image(reconstructed, caption="Original and Reconstructed Image", use_column_width=True)
