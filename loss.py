import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, masks=None):
        """
        Compute the custom loss for the multi-task learning setup.
        """
        # Ensure targets are float for BCE
        t_2_1 = targets["t_2_1"].float()
        t_2_2 = targets["t_2_2"].float()
        t_2_3 = targets["t_2_3"].float()

        # 1. Task 2.1: Binary Soft BCE Loss (Always computed)
        L1 = F.binary_cross_entropy_with_logits(
            outputs["logits_2_1"], t_2_1, reduction="none"
        )

        # 2. Task 2.2 & 2.3: Conditional Soft BCE Losses
        L2 = F.binary_cross_entropy_with_logits(
            outputs["logits_2_2"], t_2_2, reduction="none"
        )

        # For multi-label (2.3), take the mean across the 5 class dimensions
        L3 = F.binary_cross_entropy_with_logits(
            outputs["logits_2_3"], t_2_3, reduction="none"
        ).mean(dim=1, keepdim=True)

        # Conditional Confidence Weighting
        # We only want to train 2.2 and 2.3 on memes where AT LEAST 1 annotator said "YES" (t_2_1 > 0)
        is_sexist_mask = (t_2_1 > 0.0).float()

        # We multiply the loss by the probability of 2.1 (Le-Wi-Di Agreement) AND the mask
        weighting_factor = t_2_1 * is_sexist_mask

        L2_weighted = L2 * weighting_factor
        L3_weighted = L3 * weighting_factor

        # valid_count must count how many memes were actually sexist in this batch
        valid_count = is_sexist_mask.sum() + 1e-8

        # Batch averages
        loss_1 = L1.mean()
        loss_2 = L2_weighted.sum() / valid_count
        loss_3 = L3_weighted.sum() / valid_count

        # Final Total Loss
        total_loss = (loss_1 + loss_2 + loss_3)

        return {
            "loss_2_1": loss_1,
            "loss_2_2": loss_2,
            "loss_2_3": loss_3,
            "total_loss": total_loss,
        }
