import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

 # assuming the VAE model is defined in a file called vae.py

# Load the trained VAE model
model = torch.load('model.pt')
model.load_state_dict(torch.load('modelSD.pt'))
model.eval()

# Define a function to generate a random image
def generate_random_image(model):
    # Generate a random image
    image = model.sample(1)[0]

    # Convert the image to a PIL image and resize it to a displayable size
    image = transforms.ToPILImage()(image)
    image = image.resize((256, 256))

    return image

# Use Streamlit to display the generated image
st.title('CIFAR-10 VAE')
st.write('Generate a random image from the trained VAE model:')
if st.button('Generate'):
    image = generate_random_image(model)
    st.image(image)
