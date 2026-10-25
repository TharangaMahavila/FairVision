#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms

# Force Hugging Face and tqdm to use the simple command-line logger instead of notebook widgets
os.environ["TQDM_DISABLE"] = "1"

# 1. Prevent background threading hooks from multiplying open files
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def get_data_loaders(batch_size=64):
    """
    Downloads FairFace 0.25, builds training data augmentations,
    and returns production PyTorch DataLoaders.
    """
    print("Loading FairFace dataset configuration (0.25)...")
    raw_datasets = load_dataset("HuggingFaceM4/FairFace", "0.25")

    # Baseline transforms matching project parameters
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocessing formatting map hooks
    def preprocess_train(batch):
        batch["pixel_values"] = [train_transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    def preprocess_val(batch):
        batch["pixel_values"] = [val_transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    train_dataset = raw_datasets["train"].with_transform(preprocess_train)
    val_dataset = raw_datasets["validation"].with_transform(preprocess_val)

    # Custom collate collector to retain fairness tags (Race and Gender)
    def collate_fn(batch):
        images = torch.stack([x["pixel_values"] for x in batch])
        ages = torch.tensor([x["age"] for x in batch], dtype=torch.long)
        races = [x["race"] for x in batch]
        genders = [x["gender"] for x in batch]
        return {"images": images, "ages": ages, "races": races, "genders": genders}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader


# In[ ]:




