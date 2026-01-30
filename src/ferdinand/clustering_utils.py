import os, json, torch
import timm

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from PIL import Image

import numpy as np
import igraph as ig
import leidenalg
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

class FileListDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def collate_fn(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)

def extract_features(image_files, model_name = "microsoft/swin-tiny-patch4-window7-224"): # "openai/clip-vit-base-patch16"): 

    # image_files = ["img1.jpg", "img2.jpg", ...]
    # CLIP-Modell

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = FileListDataset(image_files)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    features, paths = [], []

    with torch.no_grad():
        for imgs, img_paths in dataloader:
            inputs = processor(images=imgs, return_tensors="pt").to(device)

            #embeddings = model.get_image_features(**inputs)

            outputs = model(**inputs)
            embeddings = outputs.pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            features.append(embeddings.cpu())
            paths.extend(img_paths)

    features = torch.cat(features).numpy()

    return paths, features

def build_knn_graph(features, n_neighbors=5):
    # KNN-Graph bauen
    A = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode='connectivity',
        include_self=False
    )
    A = A.maximum(A.T)  # symmetrisieren

    # igraph konstruieren
    sources, targets = A.nonzero()
    g = ig.Graph(
        n=A.shape[0],
        edges=np.column_stack([sources, targets]),
        directed=False
    )
    return g

def finetune_vitmae_imagefolder(
    data_dir: str,
    out_dir: str = "./out_sup",
    bs: int = 8,
    epochs: int = 100,
    lr: float = 3e-5,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Fine-tune a ViT-MAE model on X-ray images organized in subfolders by class.
    The data_dir should contain class-named folders, e.g.:
        data_dir/
            ok/
            uncomplete/
            false_orientation/
    The function automatically splits into train/val (80/20).
    """

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- Transforms ---
    tfm_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    # --- Load dataset and split train/val ---
    full_ds = datasets.ImageFolder(data_dir, transform=tfm_train)
    n = len(full_ds)
    n_train = int(0.8 * n)
    train_ds, val_ds = random_split(full_ds, [n_train, n - n_train], generator=torch.Generator().manual_seed(seed))
    val_ds.dataset.transform = tfm_val  # use val transform

    # --- Dataloaders ---
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # --- Class info ---
    num_classes = len(full_ds.classes)
    class_to_idx = full_ds.class_to_idx
    with open(os.path.join(out_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print("[INFO] classes:", class_to_idx)

    # --- Model ---
    model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)

    best_val = 1e9
    best_path = os.path.join(out_dir, "vitmae_sup_best.pt")

    # --- Training loop ---
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += float(loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += float(criterion(out, y))
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

        tr_loss /= max(1, len(train_dl))
        val_loss /= max(1, len(val_dl))
        acc = correct / total if total else 0.0
        print(f"[E{epoch:02d}] train {tr_loss:.4f} | val {val_loss:.4f} | acc {acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved best weights → {best_path}")

    print("[DONE] Fine-tuning finished.")
    return best_path, class_to_idx

def leiden_with_resolution(g, resolution, seed=42):
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed
    )
    modularity = partition.modularity
    n_clusters = len(np.unique(partition.membership))
    return modularity, n_clusters, partition

def find_best_resolution(
    features,
    resolutions=np.linspace(0.1, 2.0, 20),
    n_neighbors=5,
    repeats=1,
    seed=42
):

    g = build_knn_graph(features, n_neighbors=n_neighbors)

    results = []
    best_partition = None
    best_modularity = -np.inf
    best_resolution = None

    print("🔍 Suche optimalen Resolution-Parameter...\n")

    for r in tqdm(resolutions):
        modularity_scores = []

        for rep in range(repeats):
            modularity, n_clusters, partition = leiden_with_resolution(g, r, seed + rep)
            modularity_scores.append(modularity)

        mean_modularity = np.mean(modularity_scores)
        results.append((r, mean_modularity, n_clusters))

        if mean_modularity > best_modularity:
            best_modularity = mean_modularity
            best_resolution = r
            best_partition = partition

    return {
        "best_resolution": best_resolution,
        "best_modularity": best_modularity,
        "best_partition": best_partition,
        "results": results
    }