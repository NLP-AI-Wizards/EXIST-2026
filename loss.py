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
        # Using binary cross entropy with the soft marginal probabilities from the annotators
        L2_raw = F.binary_cross_entropy_with_logits(
            outputs["logits_2_2"], t_2_2, reduction="none"
        )

        # 3. Task 2.3 (Categorization - Multi-label)
        # Weighting subtasks by their frequency or importance
        pos_weight_2_3 = torch.tensor([2.0, 2.0, 3.0, 5.0, 2.0], device=t_2_3.device)
        L3_raw = F.binary_cross_entropy_with_logits(
            outputs["logits_2_3"], t_2_3, reduction="none", pos_weight=pos_weight_2_3
        )

        # Hierarchical Dependency: 
        # Task 2.2 and 2.3 should only be learned if Task 2.1 is somewhat likely.
        # We use a soft mask based on the actual annotator agreement of task 2.1.
        # If t_2_1 is 0, it means NO annotators said it was sexist.
        is_sexist_mask = (t_2_1 > 0.0).float()
        
        # Calculate mean loss for task 2.3 across the 5 categories
        L3_raw = L3_raw.mean(dim=1, keepdim=True)

        L2_masked = L2_raw * is_sexist_mask
        L3_masked = L3_raw * is_sexist_mask

        # Normalize by the number of valid (potentially sexist) samples
        valid_count = is_sexist_mask.sum() + 1e-8

        loss_2 = L2_masked.sum() / valid_count
        loss_3 = L3_masked.sum() / valid_count

        # Total Loss with Scaling
        total_loss = L1 + (self.lambda_2 * loss_2) + (self.lambda_3 * loss_3)

        return {
            "loss_2_1": L1,
            "loss_2_2": loss_2,
            "loss_2_3": loss_3,
            "total_loss": total_loss,
        }
