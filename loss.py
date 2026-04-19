import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_kl_divergence_with_logits(logits, targets, pos_weight=None):
    """
    Computes BCE with logits and subtracts the entropy of the targets
    to get the exact binary KL divergence.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )

    # Clamp to avoid log(0) instability
    p = torch.clamp(targets, 1e-7, 1.0 - 1e-7)

    # Compute target entropy depending on pos_weight presence
    if pos_weight is not None:
        entropy = -(pos_weight * p * torch.log(p) + (1 - p) * torch.log(1 - p))
    else:
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    # KL = BCE - Entropy. Use ReLU to bound floating point inaccuracies above 0.
    return F.relu(bce_loss - entropy)


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable log-variances (s = log(sigma^2))
        # Initialized to 0.0 (which means sigma = 1.0 initially)
        self.log_var_1 = nn.Parameter(torch.zeros(1))
        self.log_var_2 = nn.Parameter(torch.zeros(1))
        self.log_var_3 = nn.Parameter(torch.zeros(1))

    def forward(self, outputs, targets, masks=None):
        t_2_1 = targets["t_2_1"].float()
        t_2_2 = targets["t_2_2"].float()
        t_2_3 = targets["t_2_3"].float()

        # -----------------------------------------------------------------
        # 1. Compute KL Divergence Losses
        # -----------------------------------------------------------------
        L1_raw = binary_kl_divergence_with_logits(outputs["logits_2_1"], t_2_1)
        L2_raw = binary_kl_divergence_with_logits(outputs["logits_2_2"], t_2_2)

        # 3. Task 2.3 (Categorization - Multi-label)
        # Weighting subtasks by their frequency or importance
        pos_weight_2_3 = torch.tensor([2.0, 2.0, 3.0, 5.0, 2.0], device=t_2_3.device)
        L3_raw = binary_kl_divergence_with_logits(
            outputs["logits_2_3"], t_2_3, pos_weight=pos_weight_2_3
        ).mean(dim=1, keepdim=True)

        # -----------------------------------------------------------------
        # 2. Apply Kendall's Uncertainty Weighting (1 / 2*sigma^2)
        # exp(-log_var) is mathematically equivalent to (1 / sigma^2)
        # -----------------------------------------------------------------
        precision_1 = torch.exp(-self.log_var_1)
        precision_2 = torch.exp(-self.log_var_2)
        precision_3 = torch.exp(-self.log_var_3)

        # Base scaled losses (plus the log(sigma) regularization term)
        L1_scaled = (precision_1 * L1_raw) + (0.5 * self.log_var_1)
        L2_scaled = (precision_2 * L2_raw) + (0.5 * self.log_var_2)
        L3_scaled = (precision_3 * L3_raw) + (0.5 * self.log_var_3)

        # -----------------------------------------------------------------
        # 3. Apply the Hierarchical Probability Chain
        # -----------------------------------------------------------------
        # The mask is 1 if it's sexist, 0 if it's benign
        is_sexist_mask = (t_2_1 > 0.0).float()

        # Multiply BOTH the loss and the regularizer by the mask!
        L2_masked = L2_scaled * is_sexist_mask
        L3_masked = L3_scaled * is_sexist_mask

        # -----------------------------------------------------------------
        # 4. Aggregate
        # -----------------------------------------------------------------
        valid_count = is_sexist_mask.sum() + 1e-8

        loss_1 = L1_scaled.mean()
        loss_2 = L2_masked.sum() / valid_count
        loss_3 = L3_masked.sum() / valid_count

        total_loss = loss_1 + loss_2 + loss_3

        return {
            "loss_2_1": loss_1,
            "loss_2_2": loss_2,
            "loss_2_3": loss_3,
            "total_loss": total_loss,
            # Logging the sigmas is highly recommended to monitor task uncertainty
            "sigma_1": torch.exp(0.5 * self.log_var_1),
            "sigma_2": torch.exp(0.5 * self.log_var_2),
            "sigma_3": torch.exp(0.5 * self.log_var_3),
        }
