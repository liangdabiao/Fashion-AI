import io
import time
import base64

import requests as req
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

import config
from utils import image_to_uri, save_image


def _get_session() -> req.Session:
    session = req.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def generate_promo_photo(new_image: Image.Image,
                         ref_image: Image.Image, style_prompt: str,
                         scene_hint: str, new_id: str,
                         aspect_ratio: str = None, image_size: str = None,
                         model_name: str = None) -> list[Image.Image]:
    if aspect_ratio is None:
        aspect_ratio = config.ASPECT_RATIO
    if image_size is None:
        image_size = config.IMAGE_SIZE

    model_id = config.get_image_gen_model(model_name)

    gen_prompt = (
        f"I have a new clothing product (Image 1: flat-lay photo) and a reference "
        f"promotional photo from our bestselling catalog (Image 2).\n\n"
        f"Generate a professional e-commerce promotional photograph of a female model "
        f"wearing the clothing from Image 1.\n\n"
        f"Style guidance: {style_prompt}\n\n"
        f"Scene hint: {scene_hint}\n\n"
        f"Requirements:\n"
        f"- Full body shot, photorealistic, high quality\n"
        f"- The clothing should match Image 1 exactly\n"
        f"- The photo style and mood should match Image 2\n"
        f"- Clothing fits the body naturally, no visible tags, no extra elements"
    )

    gen_content = [
        {"type": "image_url", "image_url": {"url": image_to_uri(new_image)}},
        {"type": "image_url", "image_url": {"url": image_to_uri(ref_image)}},
        {"type": "text", "text": gen_prompt},
    ]

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": gen_content}],
        "modalities": ["image", "text"],
    }

    if "google" in model_id:
        payload["image_config"] = {"aspect_ratio": aspect_ratio, "image_size": image_size}
    elif "openai" in model_id:
        payload["image_config"] = {"aspect_ratio": aspect_ratio}

    print(f"Generating promotional photo with {model_id}...")
    for attempt in range(4):
        try:
            resp = _get_session().post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            data = resp.json()
            break
        except (req.exceptions.SSLError, req.exceptions.ConnectionError) as e:
            if attempt < 3:
                wait = (attempt + 1) * 5
                print(f"  Network error (attempt {attempt+1}/4), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    data = resp.json()
    print("Done!")

    generated_images = _extract_images(data)

    msg = data.get("choices", [{}])[0].get("message", {})
    text_content = msg.get("content", "")
    if text_content:
        print(f"Model response: {text_content[:300]}\n")

    if generated_images:
        for i, img in enumerate(generated_images):
            path = save_image(img, f"promo_{new_id}_{i+1}.png")
            print(f"Saved: {path}")
    else:
        error = data.get("error", {})
        if "unsupported_country" in str(error.get("metadata", {})):
            print("Error: OpenAI image models are not available in your region.")
            print(f"  Try using: --model nano-banana")
        else:
            print("No image generated. Raw response:")
            print(data)

    return generated_images


def _extract_images(data: dict) -> list[Image.Image]:
    images = []
    msg = data.get("choices", [{}])[0].get("message", {})
    if "images" in msg and msg["images"]:
        for img_data in msg["images"]:
            url = img_data["image_url"]["url"]
            if url.startswith("data:image"):
                b64 = url.split(",", 1)[1]
                images.append(Image.open(io.BytesIO(base64.b64decode(b64))))
            else:
                img_resp = req.get(url, timeout=30)
                images.append(Image.open(io.BytesIO(img_resp.content)))
    if not images and isinstance(msg.get("content"), list):
        for part in msg["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:image"):
                    b64 = url.split(",", 1)[1]
                    images.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                else:
                    img_resp = req.get(url, timeout=30)
                    images.append(Image.open(io.BytesIO(img_resp.content)))
    return images
