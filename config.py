import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_DIM = 2048
LLM_MODEL = "qwen/qwen3.5-397b-a17b"

IMAGE_GEN_MODELS = {
    "nano-banana": "google/gemini-3.1-flash-image-preview",
    "gpt-image": "openai/gpt-5.4-image-2",
    "gpt-image-pro": "openai/gpt-5-image",
    "gpt-image-mini": "openai/gpt-5-image-mini",
}
DEFAULT_IMAGE_GEN_MODEL = "nano-banana"

MILVUS_URI = os.environ.get("MILVUS_HOST", "")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "fashion_products")

TOP_K = 3
SALES_THRESHOLD = 1500
ASPECT_RATIO = "3:4"
IMAGE_SIZE = "2K"
EMBED_BATCH_SIZE = 5

IMAGE_DIR = "./images"
NEW_PRODUCT_DIR = "./new_products"
PRODUCT_CSV = "./products.csv"
NEW_PRODUCT_CSV = "./new_products.csv"
OUTPUT_DIR = "./output"


def get_image_gen_model(name: str = None) -> str:
    if name and name in IMAGE_GEN_MODELS:
        return IMAGE_GEN_MODELS[name]
    if name and "/" in name:
        return name
    return IMAGE_GEN_MODELS[DEFAULT_IMAGE_GEN_MODEL]
