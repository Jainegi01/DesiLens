# Cell 1: Import Libraries
!pip install diffusers["torch"] transformers
!pip install accelerate
!pip install git+https://github.com/huggingface/diffusers
!pip install peft
!pip install streamlit diffusers torch
!pip install torch

# Cell 2: Create a local tunnel
import subprocess

try:
    result = subprocess.run(['wget', '-qO-', 'https://ipv4.icanhazip.com'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    public_ipv4 = result.stdout.decode().strip()
    print(f'Public IPv4: {public_ipv4}')
except subprocess.CalledProcessError as e:
    print(f'Error: {e.stderr.decode().strip()}')

# Cell 3: Run the DesiLens Diffusion model
!streamlit run Desi_Diffusion.py & npx localtunnel --port 8501
