from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from config import *
def load_and_clean_csv(csv_path):
    df = pd.read_csv(
        csv_path,
        sep=';',
        usecols=['Scan', 'Piastra', 'n_seme', 'Time Germinazione'],
        decimal=',',
        engine='python'
    )
    print("Loaded DataFrame head:")
    print(df.head())
    print("Original shape:", df.shape)

    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df[['Scan', 'Piastra', 'n_seme', 'Time Germinazione']] = df[['Scan', 'Piastra', 'n_seme', 'Time Germinazione']].astype('int32')

    if FILTER_ZEROS:
        df = df[df['Time Germinazione'] > 0]
        print("\nFiltered out Time=0 rows.")
    else:
        print("\nKeeping Time=0 rows for dynamic labeling.")

    print("Cleaned DataFrame head:")
    print(df.head())
    print("Shape after cleaning:", df.shape)
    return df

def index_and_label_images(df_labels, images_folder, max_post_germ):
    seed_to_germ = {
        (int(r['Scan']), int(r['Piastra']), int(r['n_seme'])): int(r['Time Germinazione'])
        for _, r in df_labels.iterrows()
    }
    print(f"Loaded germination times for {len(seed_to_germ)} unique seeds.")

    image_paths = sorted(Path(images_folder).glob('*_cnn.png'))
    print(f"Found {len(image_paths)} _cnn images in {images_folder}")

    file_paths, labels = [], []
    unmatched_images, unmatched_details = [], []
    matched_count, skipped_grown = 0, 0

    for img_path in image_paths:
        filename = img_path.stem
        name = filename.replace('_cnn', '')
        parts = name.split('_')
        if len(parts) != 4:
            unmatched_images.append(filename)
            unmatched_details.append(f"Invalid part count ({len(parts)}) for {filename}")
            continue

        scan_part, mezzora_str, p_part, seed_part = parts
        try:
            scan_num = int(scan_part.replace('scan', ''))
            mezzora = int(mezzora_str)
            piastra = int(p_part.replace('p', ''))
            seed_num = int(seed_part.replace('seed', ''))
        except ValueError as e:
            unmatched_images.append(filename)
            unmatched_details.append(f"Parse error ({str(e)}) for {filename}")
            continue

        key = (scan_num, piastra, seed_num)
        germ_time = seed_to_germ.get(key)
        if germ_time is None:
            unmatched_images.append(filename)
            unmatched_details.append(f"No germination time in CSV for (Scan={scan_num}, Piastra={piastra}, n_seme={seed_num}) from {filename}")
            continue

        if germ_time == 0:
            label = 0
        elif mezzora < germ_time:
            label = 0
        elif mezzora <= germ_time + max_post_germ:
            label = 1
        else:
            skipped_grown += 1
            continue

        file_paths.append(str(img_path))
        labels.append(label)
        matched_count += 1

    if matched_count > 0:
        pos = int(np.sum(labels))
        neg = matched_count - pos
        print(f"✅ Matched and labeled {matched_count} images (including dynamic labels).")
        print(f"📊 Label counts — Germinated: {pos}, Not germinated: {neg}")

    if skipped_grown > 0:
        print(f"📊 Skipped {skipped_grown} 'too grown' images (mezzora > germ_time + {max_post_germ}).")

    if unmatched_images:
        print(f"⚠️ {len(unmatched_images)} unmatched _cnn images (e.g., {unmatched_images[:5]})")
        print("Details for first few unmatched:\n" + "\n".join(unmatched_details[:5]))

    if len(file_paths) == 0:
        raise ValueError("No matched/labeled data! Check filename units vs 'Time Germinazione'.")

    return np.array(file_paths), np.array(labels, dtype=np.int32)

def parse_seed_key(path_str):
    name = Path(path_str).stem.replace('_cnn', '')
    s, t, piastra, seed = name.split('_')
    scan = int(s.replace('scan', ''))
    piastra = int(piastra.replace('p', ''))
    seed_id = int(seed.replace('seed', ''))
    return (scan, piastra, seed_id)

def split_by_seed(paths, labels, test_size=0.2, val_size=0.2, random_state=42):
    keys = [parse_seed_key(p) for p in paths]
    seeds = sorted(set(keys))
    idx_map = defaultdict(list)
    for i, k in enumerate(keys):
        idx_map[k].append(i)

    seed_to_label = {s: int(np.max(labels[idx_map[s]])) for s in seeds}
    seeds_y = np.array([seed_to_label[s] for s in seeds], dtype=np.int32)

    seeds_train, seeds_test = train_test_split(
        seeds, test_size=test_size, random_state=random_state, stratify=seeds_y
    )
    train_y_for_strat = np.array([seed_to_label[s] for s in seeds_train], dtype=np.int32)
    seeds_train, seeds_val = train_test_split(
        seeds_train, test_size=val_size, random_state=random_state, stratify=train_y_for_strat
    )

    train_set, val_set, test_set = set(seeds_train), set(seeds_val), set(seeds_test)
    train_mask = np.array([k in train_set for k in keys])
    val_mask = np.array([k in val_set for k in keys])
    test_mask = np.array([k in test_set for k in keys])

    return (paths[train_mask], labels[train_mask],
            paths[val_mask], labels[val_mask],
            paths[test_mask], labels[test_mask])

class SeedDataset(Dataset):
    def __init__(self, paths, labels, target_size, augment=False):
        self.paths = paths
        self.labels = labels
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard per ImageNet
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(contrast=(0.9, 1.1))
        ]) if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.aug_transform:
            img = self.aug_transform(img)
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label