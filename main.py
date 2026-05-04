import argparse
import csv
import os
import sys

import numpy as np
from openai import OpenAI
from PIL import Image

import config
from data_setup import create_sample_products_csv, create_sample_new_products_csv, create_placeholder_images
from embeddings import get_image_embeddings, build_tfidf, sparse_to_dict
from milvus_store import get_client, create_collection, insert_products, hybrid_search
from style_analyzer import analyze_style
from image_generator import generate_promo_photo
from utils import load_image


def cmd_setup(args):
    """Initialize data, create Milvus collection, and insert products."""
    # Create sample data if not exists
    if not os.path.exists(config.PRODUCT_CSV):
        create_sample_products_csv()
    if not os.path.exists(config.NEW_PRODUCT_CSV):
        create_sample_new_products_csv()

    # Create placeholder images if image dirs are empty
    if not os.listdir(config.IMAGE_DIR):
        create_placeholder_images()
    if not os.listdir(config.NEW_PRODUCT_DIR):
        create_placeholder_images()

    # Load products
    with open(config.PRODUCT_CSV, newline="", encoding="utf-8") as f:
        products = list(csv.DictReader(f))

    product_images = []
    for p in products:
        img = load_image(os.path.join(config.IMAGE_DIR, p["image_path"]))
        product_images.append(img)
    print(f"Loaded {len(products)} products.")

    # Generate dense embeddings
    print("\n[1/3] Generating dense image embeddings...")
    dense_vectors = get_image_embeddings(product_images)
    print(f"Dense vectors: {dense_vectors.shape}")

    # Generate sparse embeddings
    print("\n[2/3] Building TF-IDF sparse vectors...")
    descriptions = [p["description"] for p in products]
    tfidf, sparse_vectors = build_tfidf(descriptions)
    print(f"Sparse vectors: {len(sparse_vectors)} products, vocab size: {len(tfidf.vocabulary_)}")

    # Create Milvus collection and insert
    print("\n[3/3] Creating Milvus collection and inserting data...")
    client = get_client()
    create_collection(client)
    insert_products(client, products, dense_vectors, sparse_vectors)

    print("\nSetup complete! You can now run:")
    print("  python main.py search --new-id NEW001")
    print("  python main.py generate --new-id NEW001")


def cmd_search(args):
    """Search for similar bestsellers for a new product."""
    client = get_client()

    if not client.has_collection(config.COLLECTION_NAME):
        print("Collection not found. Run 'python main.py setup' first.")
        sys.exit(1)

    # Load new products
    with open(config.NEW_PRODUCT_CSV, newline="", encoding="utf-8") as f:
        new_products = list(csv.DictReader(f))

    # Load TF-IDF model info for sparse encoding
    with open(config.PRODUCT_CSV, newline="", encoding="utf-8") as f:
        products = list(csv.DictReader(f))
    descriptions = [p["description"] for p in products]
    tfidf, _ = build_tfidf(descriptions)

    # Filter which new products to process
    if args.new_id:
        new_products = [p for p in new_products if p["new_id"] == args.new_id]
        if not new_products:
            print(f"New product '{args.new_id}' not found.")
            sys.exit(1)

    top_k = args.top_k
    threshold = args.sales_threshold

    for new_prod in new_products:
        nid = new_prod["new_id"]
        print(f"\n{'='*60}")
        print(f"Searching for: {nid}")
        print(f"Category: {new_prod['category']} | Style: {new_prod['style']}")
        print(f"{'='*60}")

        new_img = load_image(os.path.join(config.NEW_PRODUCT_DIR, new_prod["image_path"]))

        # Encode
        query_dense = get_image_embeddings([new_img], batch_size=1)[0]
        query_text = f"{new_prod['category']} {new_prod['style']} {new_prod['season']} {new_prod.get('prompt_hint', '')}"
        query_sparse = sparse_to_dict(tfidf.transform([query_text])[0])
        filter_expr = f'category == "{new_prod["category"]}" and sales_count > {threshold}'

        # Search
        results = hybrid_search(client, query_dense.tolist(), query_sparse, filter_expr, top_k)

        if not results:
            print("No similar bestsellers found.")
            continue

        print(f"\nTop-{len(results)} similar bestsellers:")
        for i, hit in enumerate(results):
            entity = hit["entity"]
            print(f"  {i+1}. {entity['product_id']} | {entity['color']} {entity['style']} "
                  f"| sales: {entity['sales_count']} | ${entity['price']:.1f} | score: {hit['distance']:.4f}")
            print(f"     {entity['description']}")


def cmd_generate(args):
    """Full pipeline: search → style analysis → image generation."""
    client = get_client()

    if not client.has_collection(config.COLLECTION_NAME):
        print("Collection not found. Run 'python main.py setup' first.")
        sys.exit(1)

    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        print("Error: Please set OPENROUTER_API_KEY in .env file.")
        sys.exit(1)

    llm = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # Load data
    with open(config.NEW_PRODUCT_CSV, newline="", encoding="utf-8") as f:
        new_products = list(csv.DictReader(f))
    with open(config.PRODUCT_CSV, newline="", encoding="utf-8") as f:
        products = list(csv.DictReader(f))
    descriptions = [p["description"] for p in products]
    tfidf, _ = build_tfidf(descriptions)

    # Filter
    if args.new_id:
        new_products = [p for p in new_products if p["new_id"] == args.new_id]
        if not new_products:
            print(f"New product '{args.new_id}' not found.")
            sys.exit(1)

    top_k = args.top_k
    threshold = args.sales_threshold
    ratio = args.aspect_ratio
    size = args.image_size
    model = args.model

    for new_prod in new_products:
        nid = new_prod["new_id"]
        print(f"\n{'='*60}")
        print(f"Processing: {nid} ({new_prod['category']}, {new_prod['style']})")
        print(f"{'='*60}")

        new_img = load_image(os.path.join(config.NEW_PRODUCT_DIR, new_prod["image_path"]))

        # Step 1: Hybrid search
        print("\n[Step 1] Hybrid search for similar bestsellers...")
        query_dense = get_image_embeddings([new_img], batch_size=1)[0]
        query_text = f"{new_prod['category']} {new_prod['style']} {new_prod['season']} {new_prod.get('prompt_hint', '')}"
        query_sparse = sparse_to_dict(tfidf.transform([query_text])[0])
        filter_expr = f'category == "{new_prod["category"]}" and sales_count > {threshold}'

        results = hybrid_search(client, query_dense.tolist(), query_sparse, filter_expr, top_k)
        if not results:
            print("No similar bestsellers found. Skipping.")
            continue

        # Load reference images
        ref_products = []
        ref_images = []
        for hit in results:
            entity = hit["entity"]
            pid = entity["product_id"]
            img = load_image(os.path.join(config.IMAGE_DIR, f"{pid}.jpg"))
            ref_products.append(entity)
            ref_images.append(img)
            print(f"  - {pid} | {entity['color']} {entity['style']} | sales: {entity['sales_count']} | score: {hit['distance']:.4f}")

        # Step 2: Style analysis
        print("\n[Step 2] Analyzing bestseller style with Qwen3.5...")
        style_prompt = analyze_style(llm, ref_images)

        # Step 3: Generate promo photo
        model_id = config.get_image_gen_model(model)
        print(f"\n[Step 3] Generating promotional photo with {model_id}...")
        scene_hint = new_prod.get("prompt_hint", "")
        generated = generate_promo_photo(
            new_img, ref_images[0], style_prompt, scene_hint, nid,
            aspect_ratio=ratio, image_size=size, model_name=model,
        )

        if generated:
            print(f"\n[{nid}] Complete! Generated {len(generated)} image(s).")
        else:
            print(f"\n[{nid}] Generation failed.")

    print(f"\nAll done! Check the {config.OUTPUT_DIR}/ directory for results.")


def main():
    parser = argparse.ArgumentParser(description="Fashion AI Promo Photo Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = subparsers.add_parser("setup", help="Initialize data and build Milvus index")
    p_setup.set_defaults(func=cmd_setup)

    # search
    p_search = subparsers.add_parser("search", help="Search similar bestsellers for new products")
    p_search.add_argument("--new-id", help="Specific new product ID (default: all)")
    p_search.add_argument("--top-k", type=int, default=config.TOP_K, help=f"Number of results (default: {config.TOP_K})")
    p_search.add_argument("--sales-threshold", type=int, default=config.SALES_THRESHOLD, help=f"Min sales count (default: {config.SALES_THRESHOLD})")
    p_search.set_defaults(func=cmd_search)

    # generate
    p_gen = subparsers.add_parser("generate", help="Full pipeline: search → analyze → generate")
    p_gen.add_argument("--new-id", help="Specific new product ID (default: all)")
    p_gen.add_argument("--top-k", type=int, default=config.TOP_K, help=f"Number of references (default: {config.TOP_K})")
    p_gen.add_argument("--sales-threshold", type=int, default=config.SALES_THRESHOLD, help=f"Min sales count (default: {config.SALES_THRESHOLD})")
    p_gen.add_argument("--aspect-ratio", default=config.ASPECT_RATIO, help=f"Image aspect ratio (default: {config.ASPECT_RATIO})")
    p_gen.add_argument("--image-size", default=config.IMAGE_SIZE, help=f"Image resolution (default: {config.IMAGE_SIZE})")
    p_gen.add_argument("--model", default=None,
                       choices=list(config.IMAGE_GEN_MODELS.keys()),
                       help=f"Image generation model (default: {config.DEFAULT_IMAGE_GEN_MODEL}). "
                            f"Options: {', '.join(config.IMAGE_GEN_MODELS.keys())}")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
