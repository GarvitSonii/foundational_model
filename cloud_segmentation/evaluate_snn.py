# evaluate_snn.py

import argparse
import pickle
import torch
from torch.utils.data import DataLoader

from cloud_segmentation.data_feeder import Cloud95Dataset
# from SNN_Unet import SpikingUNetSmall as Model  # adjust if name differs
from cloud_segmentation.Unet import UNetSmall as Model  # if you want to test ANN version

# ===== metrics (can be moved to metrics.py if you want) =====
import torch.nn.functional as F
import multiprocessing

EPS = 1e-6

def binarize(logits, thr=0.5):
    probs = torch.sigmoid(logits)
    return (probs > thr).float(), probs

def confusion_stats(pred, targets, eps=EPS):
    pred = pred.view(pred.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    TP = (pred * targets).sum(dim=1)
    FP = (pred * (1 - targets)).sum(dim=1)
    FN = ((1 - pred) * targets).sum(dim=1)
    TN = ((1 - pred) * (1 - targets)).sum(dim=1)
    return TP, FP, FN, TN

def classification_metrics_from_conf(TP, FP, FN, TN, eps=EPS):
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    acc       = (TP + TN) / (TP + TN + FP + FN + eps)
    err       = 1.0 - acc
    return precision, recall, f1, acc, err

def per_class_iou_from_conf(TP, FP, FN, TN, eps=EPS):
    iou_cloud = TP / (TP + FP + FN + eps)
    iou_clear = TN / (TN + FP + FN + eps)
    miou      = (iou_cloud + iou_clear) / 2.0
    return iou_cloud, iou_clear, miou

def dice_coeff_from_logits(logits, targets, thr=0.5, eps=EPS):
    pred, _ = binarize(logits, thr)
    inter = (pred * targets).sum(dim=(1,2,3))
    den   = pred.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice  = (2 * inter + eps) / (den + eps)
    return dice

def psnr_batch(pred_probs, targets, max_val=1.0, eps=1e-8):
    mse = F.mse_loss(pred_probs, targets, reduction='none')
    mse = mse.mean(dim=(1,2,3))
    psnr = 10.0 * torch.log10(max_val**2 / (mse + eps))
    return psnr

try:
    from skimage.metrics import structural_similarity as ssim

    def ssim_batch(pred_probs, targets):
        pred_np = pred_probs.detach().cpu().numpy()
        tgt_np  = targets.detach().cpu().numpy()
        scores = []
        for i in range(pred_np.shape[0]):
            scores.append(
                ssim(tgt_np[i,0], pred_np[i,0], data_range=1.0)
            )
        return torch.tensor(scores, dtype=torch.float32)
except ImportError:
    def ssim_batch(pred_probs, targets):
        return torch.zeros(pred_probs.size(0), dtype=torch.float32, device=pred_probs.device)

# ===== dataloader from your pkl =====

def build_val_loader(split_file, batch_size=4, num_workers=4, tilesize=512):
    with open(split_file, "rb") as f:
        split_data = pickle.load(f)

    val_items = split_data["val"]
    ds_val = Cloud95Dataset(
        val_items, 
        tilesize=tilesize, 
        augment=False
    )
    loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )
    return loader

# ===== evaluation loop =====

def evaluate(model, loader, device, timesteps=4):
    model.eval()

    all_prec = []; all_rec = []; all_f1 = []; all_acc = []; all_err = []
    all_iou_cloud = []; all_iou_clear = []; all_miou = []
    all_dice = []; all_psnr = []; all_ssim = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            # adjust timesteps/args to match your training
            # logits = model(imgs, timesteps=timesteps, return_last=False, reset=True)
            logits = model(imgs)  # for ANN version

            pred_bin, probs = binarize(logits, thr=0.5)

            TP, FP, FN, TN = confusion_stats(pred_bin, masks)
            prec_b, rec_b, f1_b, acc_b, err_b = classification_metrics_from_conf(TP, FP, FN, TN)
            iou_cloud_b, iou_clear_b, miou_b = per_class_iou_from_conf(TP, FP, FN, TN)

            dice_b = dice_coeff_from_logits(logits, masks)
            psnr_b = psnr_batch(probs, masks)
            ssim_b = ssim_batch(probs, masks)

            all_prec.append(prec_b.cpu());       all_rec.append(rec_b.cpu())
            all_f1.append(f1_b.cpu());           all_acc.append(acc_b.cpu())
            all_err.append(err_b.cpu())
            all_iou_cloud.append(iou_cloud_b.cpu())
            all_iou_clear.append(iou_clear_b.cpu())
            all_miou.append(miou_b.cpu())
            all_dice.append(dice_b.cpu())
            all_psnr.append(psnr_b.cpu())
            all_ssim.append(ssim_b.cpu())

    def cat_mean(lst): return torch.cat(lst).mean().item()

    results = {
        "Precision":   cat_mean(all_prec),
        "Recall":      cat_mean(all_rec),
        "F1-score":    cat_mean(all_f1),
        "Overall Acc": cat_mean(all_acc),
        "Error Rate":  cat_mean(all_err),
        "IoU_cloud":   cat_mean(all_iou_cloud),
        "IoU_clear":   cat_mean(all_iou_clear),
        "mIoU":        cat_mean(all_miou),
        "Dice":        cat_mean(all_dice),
        "PSNR":        cat_mean(all_psnr),
        "SSIM":        cat_mean(all_ssim),
    }
    return results

# ===== main =====

def main():
    torch.set_num_threads(multiprocessing.cpu_count())  # e.g., 16
    torch.set_num_interop_threads(4)
    torch.set_float32_matmul_precision('medium')
    print("Starting evaluation script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt model")
    parser.add_argument("--split_file", required=True, help="train_val_split.pkl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--timesteps", type=int, default=6)  # match your training TIMESTEPS
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint}")
    model = Model()  # add channels/params if needed
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # maybe checkpoint is raw state_dict
        state_dict = ckpt

    model.load_state_dict(state_dict)




    
    model.to(device)

    print(f"Loading val split from: {args.split_file}")
    loader = build_val_loader(args.split_file,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              tilesize=128)  # match your training tilesize

    print("Evaluating...")
    results = evaluate(model, loader, device, timesteps=args.timesteps)

    print("\n=== Evaluation Results (val set) ===")
    for k, v in results.items():
        if k == "PSNR":
            print(f"{k:12s}: {v:.2f}")
        else:
            print(f"{k:12s}: {v:.4f}")

if __name__ == "__main__":
    main()
