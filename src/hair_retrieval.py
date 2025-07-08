import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from models import models_vit 
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse


# --------- ARGUMENT PARSER ---------
def parse_args():
    parser = argparse.ArgumentParser(description='Hair Image Retrieval using Vision Transformer')
    
    # Model configuration
    parser.add_argument('--ckpt_path', type=str, default="checkpoint/checkpoint-199.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default="vit_base_patch16",
                        choices=["vit_base_patch16", "sim_vit_base_patch16"],
                        help='Model architecture to use')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, 
                        default="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train",
                        help='Path to training data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    
    # Device and output configuration
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda/cpu). If None, auto-detect')
    parser.add_argument('--embed_save_dir', type=str, default="save/embeddings",
                        help='Directory to save embeddings')
    
    # Retrieval configuration
    parser.add_argument('--query_image', type=str, default=None,
                        help='Path to query image for retrieval (if None, use first image from dataset)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top similar images to retrieve')
    
    # Action flags
    parser.add_argument('--extract_only', action='store_true',
                        help='Only extract embeddings, skip retrieval')
    parser.add_argument('--retrieve_only', action='store_true',
                        help='Only perform retrieval, skip embedding extraction')
    parser.add_argument('--force_extract', action='store_true',
                        help='Force re-extraction of embeddings even if they exist')
    
    return parser.parse_args()

# --------- CONFIG ---------
args = parse_args()

# Set device
if args.device is None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = args.device

# Create save directory
os.makedirs(args.embed_save_dir, exist_ok=True)
# --------------------------

# --------- MODEL DEFINITION ---------

def build_model():
    model = models_vit.__dict__[args.model_name](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=True,
        init_values=None,
    )
    return model

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Model loading message:", msg)
    return model
# ------------------------------------

# --------- FEATURE EXTRACTOR ---------
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    def extract_features(self, x):
        with torch.no_grad():
            features = self.model.forward_features(x)
            return features[:, 0]  # CLS token
# -------------------------------------

# --------- DATASET & TRANSFORM ---------
transform = transforms.Compose([
    transforms.Resize(224, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# ---------------------------------------

# --------- MAIN INFERENCE ---------
def extract_and_save_embeddings():
    print(f"Loading dataset from: {args.data_path}")
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Building model: {args.model_name}")
    model = build_model()
    model = load_checkpoint(model, args.ckpt_path)
    model.to(DEVICE)
    model.eval()
    feature_extractor = FeatureExtractor(model)

    all_embeddings = []
    all_paths = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(DEVICE)
            features = feature_extractor.extract_features(imgs)
            all_embeddings.append(features.cpu().numpy())
            start_idx = len(all_paths)
            all_paths.extend([dataset.samples[i][0] for i in range(start_idx, start_idx + imgs.size(0))])
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(os.path.join(args.embed_save_dir, "embeddings.npy"), all_embeddings)
    with open(os.path.join(args.embed_save_dir, "image_paths.txt"), "w") as f:
        for path in all_paths:
            f.write(path + "\n")
    print(f"Saved {all_embeddings.shape[0]} embeddings and paths to {args.embed_save_dir}")

# --------- RETRIEVAL UTILS ---------
def load_embeddings_and_paths():
    embeddings = np.load(os.path.join(args.embed_save_dir, "embeddings.npy"))
    with open(os.path.join(args.embed_save_dir, "image_paths.txt"), "r") as f:
        paths = [line.strip() for line in f.readlines()]
    return embeddings, paths

def check_embeddings_exist():
    """Check if embeddings and paths files already exist"""
    embeddings_file = os.path.join(args.embed_save_dir, "embeddings.npy")
    paths_file = os.path.join(args.embed_save_dir, "image_paths.txt")
    return os.path.exists(embeddings_file) and os.path.exists(paths_file)

def retrieve_similar_images(query_embedding, all_embeddings, all_paths, top_k=5):
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({'path': all_paths[idx], 'similarity': similarities[idx]})
    return results

def process_query_image(image_path):
    print(f"Processing query image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    model = build_model()
    model = load_checkpoint(model, args.ckpt_path)
    model.to(DEVICE)
    model.eval()
    feature_extractor = FeatureExtractor(model)
    embedding = feature_extractor.extract_features(image_tensor).cpu().numpy()[0]
    return embedding

# --------- MAIN ---------
if __name__ == "__main__":
    print("=" * 60)
    print("HAIR IMAGE RETRIEVAL SYSTEM")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Checkpoint: {args.ckpt_path}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Device: {DEVICE}")
    print(f"  - Embed save dir: {args.embed_save_dir}")
    print(f"  - Top K: {args.top_k}")
    print("=" * 60)
    
    # Check if we should extract embeddings
    should_extract = not args.retrieve_only and (args.force_extract or not check_embeddings_exist())
    
    if should_extract:
        if args.force_extract:
            print("Force extraction enabled. Extracting embeddings...")
        else:
            print("Embeddings not found. Extracting embeddings for all images...")
        extract_and_save_embeddings()
    else:
        if not args.extract_only:
            print("Embeddings already exist. Loading from disk...")
    
    # Check if we should perform retrieval
    if not args.extract_only:
        print("\n" + "=" * 60)
        print("PERFORMING IMAGE RETRIEVAL")
        print("=" * 60)
        
        embeddings, paths = load_embeddings_and_paths()
        
        # Determine query image
        if args.query_image:
            query_img_path = args.query_image
        else:
            query_img_path = paths[0]
            print(f"No query image specified, using first image from dataset: {query_img_path}")
        
        query_embedding = process_query_image(query_img_path)
        results = retrieve_similar_images(query_embedding, embeddings, paths, top_k=args.top_k)
        
        print(f"\nTop {args.top_k} similar images to: {query_img_path}")
        print("-" * 60)
        for i, res in enumerate(results):
            print(f"{i+1}. {res['path']} (similarity: {res['similarity']:.4f})")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)