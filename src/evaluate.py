import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Handle imports
from src.utils import Config, ssim, post_process, get_sobel_map
from src.dataset import MVTecDataset
from src.models import Autoencoder, PredictorFCN, ResNetBackbone

def denormalize(img_tensor):
    """Convert tensor to numpy for visualization [0, 1] -> [0, 255]"""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    # Scale to 0-255
    img = (img * 255).astype(np.uint8)
    return img

def visualize_results(image_tensor, template_tensor, reconstructed_tensor, gt_mask, raw_score_map, processed_mask, save_path, label, pred_score):
    """
    Tạo bảng so sánh trực quan với các lớp phủ (overlay) giúp dễ dàng quan sát:
    1. Ảnh gốc với đường bao Ground Truth (Màu xanh lá)
    2. Golden Template (Ảnh mẫu chuẩn)
    3. Ảnh tái tạo bởi Autoencoder (Dùng để so sánh sự khác biệt)
    4. Bản đồ nhiệt (Heatmap) dự đoán lỗi
    5. Ảnh gốc chồng lớp Heatmap (Overlay)
    6. Ảnh gốc chồng lớp Mask dự đoán (Overlay - Màu đỏ)
    """
    img = denormalize(image_tensor[0])
    template = denormalize(template_tensor[0])
    reconstructed = denormalize(reconstructed_tensor[0])
    
    # Xử lý GT Mask
    gt_mask_np = (gt_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    
    # Chuyển đổi Score Map thành Heatmap màu
    score_map_norm = ((raw_score_map - raw_score_map.min()) / (raw_score_map.max() - raw_score_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Tạo các lớp phủ (Overlays)
    # 1. Overlay Heatmap
    alpha = 0.4
    overlay_heatmap = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    # 2. Overlay Dự đoán (Đường bao màu đỏ)
    overlay_pred = img.copy()
    contours, _ = cv2.findContours(processed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_pred, contours, -1, (255, 0, 0), 2)
    
    # 3. Overlay Ground Truth (Đường bao màu xanh lá)
    overlay_gt = img.copy()
    gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_gt, gt_contours, -1, (0, 255, 0), 2)

    status = "ANOMALOUS" if label == 1 else "NORMAL"
    pred_status = "DEFECT DETECTED" if pred_score > 0.5 else "PASS"
    color = "red" if label == 1 else "green"

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Anomaly Detection Report | GT: {status} | Pred: {pred_status} (Score: {pred_score:.4f})", 
                 fontsize=18, color=color, fontweight='bold')

    # Hàng 1: Các ảnh cơ bản
    axes[0, 0].imshow(overlay_gt)
    axes[0, 0].set_title("1. Original + GT (Green)", fontsize=12)
    
    axes[0, 1].imshow(template)
    axes[0, 1].set_title("2. Golden Template", fontsize=12)
    
    axes[0, 2].imshow(reconstructed)
    axes[0, 2].set_title("3. AE Reconstruction", fontsize=12)
    
    axes[0, 3].imshow(heatmap)
    axes[0, 3].set_title("4. Raw Anomaly Heatmap", fontsize=12)

    # Hàng 2: Các lớp phủ phân tích
    axes[1, 0].imshow(overlay_heatmap)
    axes[1, 0].set_title("5. Overlay: Image + Heatmap", fontsize=12)
    
    axes[1, 1].imshow(score_map_norm, cmap='magma')
    axes[1, 1].set_title("6. Anomaly Intensity Map", fontsize=12)
    
    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title("7. Overlay: Image + Prediction (Red)", fontsize=12)
    
    axes[1, 3].imshow(processed_mask, cmap='gray')
    axes[1, 3].set_title("8. DIP Refined Mask (Final)", fontsize=12)

    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=120)
    plt.close()

def evaluate(category):
    device = torch.device(Config.DEVICE)
    print(f"\n[EVALUATION] Starting evaluation for: {category}")
    
    dataset = MVTecDataset(category, split='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    ae = Autoencoder().to(device)
    predictor = PredictorFCN().to(device)
    backbone = ResNetBackbone().to(device).eval()
    
    ae_path = f"checkpoints/ae_{category}.pth"
    pred_path = f"checkpoints/pred_{category}.pth"
    temp_path = f"checkpoints/template_{category}.pth"
    
    if not os.path.exists(ae_path) or not os.path.exists(pred_path) or not os.path.exists(temp_path):
        print(f"!!! Skipping {category}: Checkpoints not found.")
        return

    ae.load_state_dict(torch.load(ae_path, map_location=device))
    predictor.load_state_dict(torch.load(pred_path, map_location=device))
    template = torch.load(temp_path, map_location=device)
    
    ae.eval()
    predictor.eval()
    
    pixel_preds, pixel_masks = [], []
    image_labels, image_scores = [], []
    
    result_dir = f"results/{category}"
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Processing {len(dataloader)} images...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].numpy().flatten()
            label = batch['label'].item()
            
            resnet_features = backbone(images)
            reconstructed = ae(images)
            
            # Tính toán sai số tái tạo và sai số so với template
            res_ae = torch.abs(images - reconstructed).mean(dim=1, keepdim=True)
            res_temp = torch.abs(images - template).mean(dim=1, keepdim=True)
            residual_map = 0.5 * res_ae + 0.5 * res_temp
            
            # Dự đoán vùng bất thường
            pred_logits = predictor(residual_map, resnet_features)
            pred_map = torch.sigmoid(pred_logits)
            
            # Kết hợp với SSIM để tăng độ chính xác cấu trúc
            ssim_val_map = 1.0 - ssim(images, reconstructed, size_average=False)
            combined_map = pred_map * ssim_val_map
            combined_np = combined_map[0, 0].cpu().numpy()
            
            # Hậu xử lý bằng các thuật toán xử lý ảnh truyền thống (DIP)
            processed_mask = post_process(combined_np) 
            
            # Tính điểm số cho toàn bộ bức ảnh (Image-level score)
            # Nếu có vùng lỗi sau hậu xử lý, lấy max vùng đó. Nếu không lấy trung bình raw.
            if np.any(processed_mask):
                img_score = np.max(combined_np * (processed_mask > 0))
            else:
                img_score = np.mean(combined_np)
            
            pixel_preds.extend(combined_np.flatten())
            pixel_masks.extend(masks)
            image_labels.append(label)
            image_scores.append(img_score)
            
            # Lưu ảnh minh họa (Lưu tất cả ảnh lỗi và 5 ảnh bình thường đầu tiên)
            if (label == 1) or (i < 5):
                img_type = "anomaly" if label == 1 else "normal"
                save_path = f"{result_dir}/{img_type}_{i:03d}.png"
                visualize_results(
                    images, template, reconstructed, batch['mask'], 
                    combined_np, processed_mask, save_path, label, img_score
                )

    # Tính toán các chỉ số đánh giá
    pixel_masks_bin = (np.array(pixel_masks) > 0.5).astype(np.int32)
    pixel_preds_np = np.array(pixel_preds)
    
    pixel_auc = roc_auc_score(pixel_masks_bin, pixel_preds_np)
    image_auc = roc_auc_score(image_labels, image_scores)
    
    print(f"\n[RESULTS] {category.upper()}")
    print(f"  - Pixel-level AUROC: {pixel_auc:.4f}")
    print(f"  - Image-level AUROC: {image_auc:.4f}")
    print(f"  - Reports saved in:  {result_dir}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate("wood")
