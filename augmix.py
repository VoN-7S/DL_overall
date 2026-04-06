"""
CutMix Augmentation
===================
CutMix mixes two training samples by cutting a rectangular patch from image B
and pasting it onto image A, then mixing the labels proportionally to the area:

    ỹ = λ · y_A  +  (1 - λ) · y_B
    L = λ · CE(f(x̃), y_A)  +  (1-λ) · CE(f(x̃), y_B)

Why it helps
------------
• Unlike Cutout (which zero-fills), every pixel still carries real signal.
• Strong regularizer — forces the model to use the whole image, not just the
  most discriminative patch.
• Improves calibration and often outperforms Mixup on natural images.

Reference: Yun et al., "CutMix: Training Strategy that Makes Use of Sample
Pairings," ICCV 2019.  https://arxiv.org/abs/1905.04899
"""

# ── Core ──────────────────────────────────────────────────────────────────────

def rand_bbox(H, W, lam):
    """
    Sample a random box whose area ≈ (1-lam)·H·W.

    Box size derivation
    -------------------
    cut_ratio = sqrt(1 - lam)  ensures  area(box)/area(image) ≈ 1 - lam.
    Center is sampled uniformly; edges are clipped to image boundaries.
    lam is recomputed from the actual clipped area so the label mix is exact.
    """
    r = np.sqrt(1.0 - lam)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - int(W*r)//2, 0, W), np.clip(cx + int(W*r)//2, 0, W)
    y1, y2 = np.clip(cy - int(H*r)//2, 0, H), np.clip(cy + int(H*r)//2, 0, H)
    return x1, y1, x2, y2, 1.0 - (x2-x1)*(y2-y1)/(H*W)


def cutmix_criterion(criterion, outputs, metadatas):
    """
    CutMix-aware loss: weighted sum of two cross-entropy terms.
    lam is the fraction of image A retained, so it weights CE toward y_a.
    """
    y_a = torch.tensor([metadata["target_label"] for metadata in metadatas])
    y_b = torch.tensor([metadata["source_label"] for metadata in metadatas])
    lam = torch.tensor([metadata["target_ratio"] for metadata in metadatas])
    lam = lam.to(outputs.device)
    return (lam * criterion(outputs, y_a.to(outputs.device)) +
            (1 - lam) * criterion(outputs, y_b.to(outputs.device))).mean()


import torch
import numpy as np

def cutmix(targets, target_labels, source, source_labels):
    """
    targets: Images to be pasted ONTO
    target_labels: Labels of targets
    source: Images to take patches FROM
    source_labels: Labels of source images
    """
    height, width = targets.shape[-2:]
    new_images = targets.clone()
    metadata = []

    for i in range(targets.shape[0]):
        # 1. Sample mixing ratio and bounding box
        # Standard CutMix uses a Beta distribution, but we'll use your uniform logic
        lamda = np.random.uniform(0.1, 0.9) 
        x1, y1, x2, y2, area = rand_bbox(height, width, lamda)

        # 2. Pick a random source image
        idx = torch.randint(0, source.shape[0], (1,)).item()
        source_img = source[idx]
        s_label = source_labels[idx].item()
        t_label = target_labels[i].item()

        # 3. Apply the patch
        new_images[i, :, y1:y2, x1:x2] = source_img[:, y1:y2, x1:x2]

        # 4. Calculate exact pixel-based lambda (more accurate than the sampled one)
        actual_lamda = 1.0 - (float((x2 - x1) * (y2 - y1)) / (height * width))

        # 5. Save metadata: [target_label, source_label, target_ratio]
        # target_ratio is 'how much of the original target is left'
        metadata.append({
            "target_label": t_label,
            "source_label": s_label,
            "target_ratio": actual_lamda
        })

    return new_images, metadata

def augmixer(imgs, labels):
    unique_labels = torch.unique(labels)
    all_mixed_images = []
    all_metadata = []

    for i in range(len(unique_labels)):
        # Define target group (images of the current class)
        target_mask = (labels == unique_labels[i])
        target_imgs = imgs[target_mask]
        target_lbls = labels[target_mask]

        # Define source group (everything else)
        source_mask = (labels != unique_labels[i])
        source_imgs = imgs[source_mask]
        source_lbls = labels[source_mask]

        # Only run if we have sources to pull from
        if source_imgs.shape[0] > 0:
            mixed_imgs, metadata = cutmix(target_imgs, target_lbls, source_imgs, source_lbls)
            
            all_mixed_images.append(mixed_imgs)
            all_metadata.extend(metadata)

    # Combine all batches into one tensor
    final_images = torch.cat(all_mixed_images, dim=0)
    
    return final_images, all_metadata


