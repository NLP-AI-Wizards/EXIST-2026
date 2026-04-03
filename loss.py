import torch.nn as nn
import torch.nn.functional as F

class DISCERNLoss(nn.Module):
    def __init__(
        self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0,
    ):
        """
        Masked MTL MSE Loss + InfoNCE Contrastive Loss for DISCERN.

        Args:
            lambda_1: Weight for the base MSE loss on 2.1
            lambda_2: Weight for the conditional MSE loss on 2.2
            lambda_3: Weight for the conditional MSE loss on 2.3
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, outputs, targets, masks):
        """
        outputs: Dict from DISCERN forward pass
        targets: Dict containing the Le-Wi-Di marginal probability targets
        masks: Dict containing the conditional masking (True if N_yes > 0) and physio mask
        """
        # MTL MSE Loss
        # Base loss (Always computed)
        L1 = F.mse_loss(outputs["preds_2_1"], targets["t_2_1"], reduction="none")

        # Conditional losses
        L2 = F.mse_loss(outputs["preds_2_2"], targets["t_2_2"], reduction="none")
        L3 = F.mse_loss(outputs["preds_2_3"], targets["t_2_3"], reduction="none").mean(
            dim=1, keepdim=True
        )

        # Apply the confidence weighting (target of 2.1) + Conditional Mask
        weighting_factor = targets["t_2_1"] * masks["cond_mask"].float()

        L2_weighted = L2 * weighting_factor
        L3_weighted = L3 * weighting_factor

        # Batch averages
        valid_count = masks["cond_mask"].float().sum() + 1e-8
        loss_L1 = L1.mean()
        loss_L2 = L2_weighted.sum() / valid_count
        loss_L3 = L3_weighted.sum() / valid_count
        loss_mtl = self.lambda_1 * loss_L1 + self.lambda_2 * loss_L2 + self.lambda_3 * loss_L3

        # Final Total Loss
        total_loss = loss_mtl

        return {
            "MSE_L1": loss_L1,
            "MSE_L2": loss_L2,
            "MSE_L3": loss_L3,
            "MSE_MTL": loss_mtl,
            "total_loss": total_loss,
        }
