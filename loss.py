import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # We increase the weight for subtasks because they have fewer samples per batch
        self.lambda_2 = 2.0
        self.lambda_3 = 5.0  # Task 2.3 is the hardest; give it the loudest voice

    def forward(self, outputs, targets, masks=None):
        t_2_1 = targets["t_2_1"].float()
        t_2_2 = targets["t_2_2"].float()
        t_2_3 = targets["t_2_3"].float()

        # 1. Task 2.1 (Base Task)
        L1 = F.binary_cross_entropy_with_logits(
            outputs["logits_2_1"], t_2_1, reduction="mean"
        )

        # 2. Task 2.2 (Source Intention)
        L2_raw = F.binary_cross_entropy_with_logits(
            outputs["logits_2_2"], t_2_2, reduction="none"
        )

        # 3. Task 2.3 (Categorization - Multi-label)
        # We add pos_weight to prevent the model from just predicting all zeros.
        # 5.0 is a good starting point for the rare EXIST categories.
        pos_weight_2_3 = torch.full([5], 5.0).to(t_2_3.device)
        L3_raw = F.binary_cross_entropy_with_logits(
            outputs["logits_2_3"], t_2_3, reduction="none", pos_weight=pos_weight_2_3
        ).mean(dim=1, keepdim=True)

        # --- REVISED MASKING LOGIC ---
        is_sexist_mask = (t_2_1 > 0.0).float()

        # We REMOVE the t_2_1 multiplier here.
        # Even if only 1/6 people said YES, the subtask gradient should be FULL strength
        # so the model actually learns the features of that category.
        L2_masked = L2_raw * is_sexist_mask
        L3_masked = L3_raw * is_sexist_mask

        valid_count = is_sexist_mask.sum() + 1e-8

        loss_2 = L2_masked.sum() / valid_count
        loss_3 = L3_masked.sum() / valid_count

        # Total Loss with Lambda Scaling
        total_loss = L1 + (self.lambda_2 * loss_2) + (self.lambda_3 * loss_3)

        return {
            "loss_2_1": L1,
            "loss_2_2": loss_2,
            "loss_2_3": loss_3,
            "total_loss": total_loss,
        }
