import cv2
import numpy as np
import torch
import torch.nn as nn
import random
from torchvision import models, transforms
from pathlib import Path

def extract_frames_from_video(video_path, max_frames=None):
    """
    Extract frames from a video using OpenCV.
    
    Args:
        video_path (str): Path to the input video file.
        max_frames (int, optional): If set, will only read up to 'max_frames' frames 
                                    (spaced evenly) from the video to reduce computation.
    
    Returns:
        frames (list of np.ndarray): List of frames read from the video in BGR format.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    if not max_frames or max_frames > total_frames:
        max_frames = total_frames
    
    # Indices of frames to grab (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=np.int32)
    frame_set = set(frame_indices)
    
    idx = 0
    grabbed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx in frame_set:
            frames.append(frame)
            grabbed_count += 1
            if grabbed_count >= max_frames:
                break
        
        idx += 1
    
    cap.release()
    return frames

def build_resnet50_feature_extractor(device="cpu"):
    """
    Build a ResNet-50 model (pretrained on ImageNet) that outputs 2048-d feature vectors.
    
    Args:
        device (str): 'cpu' or 'cuda' (if CUDA is available).
    
    Returns:
        model (nn.Module): Modified ResNet50 (output dim: 2048).
    """
    # Load a pretrained ResNet-50
    model = models.resnet50(pretrained=True)
    # Remove the classification (FC) layer. The last layer before FC has shape (batch, 2048).
    # By slicing off the last layer, we get a feature extractor that outputs a shape of (batch, 2048, 1, 1).
    model = nn.Sequential(*list(model.children())[:-1])  # remove linear layer
    
    model.eval()
    model.to(device)
    
    return model

def compute_resnet50_embeddings(frames, model, device="cpu"):
    """
    Compute 2048-d embeddings for a list of frames (BGR images from OpenCV).
    
    Args:
        frames (list of np.ndarray): Frames from the video (BGR).
        model (nn.Module): ResNet50 feature extractor.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        embeddings (np.ndarray): Array of shape (num_frames, 2048).
    """
    
    # Define the same transforms that ImageNet models expect
    # Note: We first convert BGR -> RGB for consistency
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalization values for ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    embeddings_list = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform to tensor
        tensor_img = transform(frame_rgb).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            feat = model(tensor_img)  # shape: (1, 2048, 1, 1)
        
        # Flatten to (2048,)
        feat = feat.view(feat.size(0), -1)  # (1, 2048)
        emb = feat.squeeze(0).cpu().numpy() # get rid of batch dim => (2048,)
        
        embeddings_list.append(emb)
    
    # Return an array of shape (num_frames, 2048)
    return np.array(embeddings_list)

def select_maximally_dissimilar_frames(embeddings, N):
    """
    Greedy selection of N indices from embeddings to maximize pairwise distance.
    
    Args:
        embeddings (np.ndarray): (num_frames, 2048) array of embeddings.
        N (int): Number of frames to select.
    
    Returns:
        selected_indices (list of int): Indices of the chosen frames.
    """
    num_frames = len(embeddings)
    if N >= num_frames:
        return list(range(num_frames))
    
    # 1) Randomly pick one frame to start
    random_idx = random.randint(0, num_frames - 1)
    selected_indices = [random_idx]
    
    # 2) Iteratively pick frames that maximize sum of distances to already-selected frames
    while len(selected_indices) < N:
        best_idx = None
        best_dist_sum = -1
        
        for i in range(num_frames):
            if i in selected_indices:
                continue
            
            # Sum of distances from embeddings[i] to each selected embedding
            dist_sum = 0.0
            for sel_idx in selected_indices:
                dist_sum += np.linalg.norm(embeddings[i] - embeddings[sel_idx])
            
            if dist_sum > best_dist_sum:
                best_dist_sum = dist_sum
                best_idx = i
        
        selected_indices.append(best_idx)
    
    return selected_indices

def main():
    video_path = input("Enter the path to the video file: ")
    save_path = Path(input("Enter the path to save the output frames: "))
    N = 32                             # Number of frames to select
    max_frames_to_process = 200        # Limit how many frames to read for embedding
    
    # You can change to 'cuda' if you have a GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Extract frames from the video
    frames = extract_frames_from_video(video_path, max_frames=max_frames_to_process)
    print(f"Extracted {len(frames)} frames from {video_path}")
    
    # 2. Build a ResNet50 feature extractor
    model = build_resnet50_feature_extractor(device=device)
    
    # 3. Compute embeddings
    embeddings = compute_resnet50_embeddings(frames, model, device=device)
    print("Embeddings shape:", embeddings.shape)  # (max_frames_to_process, 2048)
    
    # 4. Select N frames that are maximally dissimilar
    chosen_indices = select_maximally_dissimilar_frames(embeddings, N)
    print("Chosen frame indices:", chosen_indices)
    
    # 5. Save or display chosen frames
    for i, idx in enumerate(chosen_indices):
        chosen_frame = frames[idx]
        out_name = f"output_{i}.jpg"
        cv2.imwrite(save_path / out_name, chosen_frame)
        print(f"Saved frame {idx} as {out_name}")

if __name__ == "__main__":
    main()