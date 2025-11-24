import torch
import torch.nn.functional as F

EPS = 1e-6

def binarize(logits, thr=0.5):
    """logits -> (hard mask, prob map), both in [0,1]."""
    probs = torch.sigmoid(logits)
    return (probs > thr).float(), probs

def confusion_stats(pred, targets, eps=EPS):
    """
    pred, targets: (N,1,H,W) in {0,1}
    returns per-sample TP, FP, FN, TN
    """
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
    """
    IoU_cloud: positive class (cloud)
    IoU_clear: negative class (non-cloud / background)
    """
    iou_cloud = TP / (TP + FP + FN + eps)
    iou_clear = TN / (TN + FP + FN + eps)
    miou      = (iou_cloud + iou_clear) / 2.0
    return iou_cloud, iou_clear, miou

def dice_coeff_from_logits(logits, targets, thr=0.5, eps=EPS):
    pred, _ = binarize(logits, thr)
    inter = (pred * targets).sum(dim=(1,2,3))
    den   = pred.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice  = (2 * inter + eps) / (den + eps)
    return dice  # per-sample

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
        # fallback if scikit-image is not installed
        return torch.zeros(pred_probs.size(0), dtype=torch.float32, device=pred_probs.device)
