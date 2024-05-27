import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
import base64
from io import BytesIO

st.set_page_config(page_title="DesiLens")

def generate_image(prompt, h, w, steps, guidance, denoising_strength, neg, lora_weight, lora_paths):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "Yntec/epiCPhotoGasm", torch_dtype=torch.float16
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(31)

    for lora_path in lora_paths:
        pipeline.load_lora_weights(lora_path)

    images = pipeline(
        prompt,
        strength=denoising_strength,
        cross_attention_kwargs={"scale": lora_weight},
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        negative_prompt=neg,
    ).images[0]

    return images

def app():
    st.title("DesiLens")
    st.markdown(
        """
         <style>
        [data-testid='stAppViewContainer'] {
            background: linear-gradient(to bottom, #FF671F, #FFFFFF, #046A38);
        }

        [data-testid='stWidgetLabel'] p{
          font-weight:700;
          font-size:18px;
        }

        [data-testid='stVerticalBlock']{
          background:rgba(255,255,255,0.7);
          padding-left:20px;
          padding-right:20px;
          padding-top:10px;
          padding-bottom:20px;
          border-radius:15px;
        }
        [data-testid='stHeader']{
          background-color:#FF671F;
        }

        [data-testid='baseButton-secondary']{
          background-color: navy;
          color: white;
        }

        @media (min-width: 768px) {
           [data-testid='textInputRootElement']{
          width:95%
        }
        [data-testid='stNumberInputContainer']{
          width:95%
          }
          [data-baseweb='select']{
          width:95%
          }
        }

        @media (max-width: 767px) {
          [data-testid='stNumberInputContainer']{
          width:90%
          }

        [data-testid='textInputRootElement']{
          width:90%
        }

        [data-baseweb='select']{
          width:90%
          }
        }

       </style>
        """,
        unsafe_allow_html=True,
    )

    prompt = st.text_input("Enter your prompt")

    states = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Gujarat", "Haryana", "Kerala", "Maharasthra", "Manipur", "Meghalaya", "Mizoram", "Punjab", "Rajasthan"]
    selected_state = st.selectbox("Select a state", states)

    gender_options = ["Man", "Woman"]
    selected_gender = st.radio("Select a gender", options=gender_options)

    h = st.number_input("Height", value=600)
    w = st.number_input("Width", value=600)
    steps = st.number_input("Number of steps", value=25)
    guidance = st.number_input("Guidance scale", value=7.0)
    denoising_strength = st.number_input("Denoising strength", value=0.4)
    lora_weight = st.number_input("Lora weight", value=0.7)

    neg = "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,"

    lora_paths = {
        "Andhra Pradesh": {
            "Man": ["jainegi/Andhra-Man"],
            "Woman": ["jainegi/Andhra-Woman"]
        },
        "Arunachal Pradesh": {
            "Man": ["jainegi/Arunachal_Man"],
            "Woman": ["jainegi/Arunachal_Woman"]
        },
        "Assam": {
            "Man": ["jainegi/Assam_Man"],
            "Woman": ["jainegi/Assam_Woman"]
        },
        "Gujarat": {
            "Man": ["jainegi/Gujarat_Man"],
            "Woman": ["jainegi/Gujarat_Woman"]
        },
        "Haryana": {
            "Man": ["jainegi/Haryana_Man"],
            "Woman": ["jainegi/Haryana_Woman"]
        },
        "Kerala": {
            "Man": ["jainegi/Kerala_Man"],
            "Woman": ["jainegi/Kerala_Woman"]
        },
        "Maharasthra": {
            "Man": ["jainegi/Maharashtra_Man"],
            "Woman": ["jainegi/Maharashtra_Woman"]
        },
        "Manipur": {
            "Man": ["jainegi/Manipur_Man"],
            "Woman": ["jainegi/Manipur_Woman"]
        },
        "Meghalaya": {
            "Man": ["jainegi/Meghalaya_Man"],
            "Woman": ["jainegi/Meghalaya_Woman"]
        },
        "Mizoram": {
            "Man": ["jainegi/Mizoram_Man"],
            "Woman": ["jainegi/Mizoram_Woman"]
        },
        "Punjab": {
            "Man": ["jainegi/Punjab_Man"],
            "Woman": ["jainegi/Punjab_Woman"]
        },
        "Rajasthan": {
            "Man": ["jainegi/Rajasthan_Man"],
            "Woman": ["jainegi/Rajasthan_Woman"]
        }
    }

    if st.button("Generate Image", help="Click to generate the image"):
        selected_lora_paths = lora_paths[selected_state][selected_gender]

        image = generate_image(prompt, h, w, steps, guidance, denoising_strength, neg, lora_weight, selected_lora_paths)
        st.image(image, caption="Generated Image")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        byte_image = buffer.getvalue()
        download_button = st.download_button(
            label="Download Image",
            data=byte_image,
            file_name="generated_image.png",
            mime="image/png",
            key="download-image",
        )

if __name__ == "__main__":
    import subprocess

    localtunnel_process = subprocess.Popen(
        ['npx', 'localtunnel', '--port', '8501'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for line in iter(localtunnel_process.stdout.readline, b''):
        line = line.decode().strip()
        if line.startswith('your url is:'):
            public_url = line.split(':')[2].strip()
            print(f"Public URL: {public_url}")
            break

    app()
