import torch
import torch.nn as nn
import torch.nn.functional as F

class DISCERNLoss(nn.Module):
    def __init__(
        self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, alpha=0.1, temperature=0.07
    ):
        """
        Masked MTL MSE Loss + InfoNCE Contrastive Loss for DISCERN.

        Args:
            lambda_1: Weight for the base MSE loss on 2.1
            lambda_2: Weight for the conditional MSE loss on 2.2
            lambda_3: Weight for the conditional MSE loss on 2.3
            alpha: Weight for the InfoNCE loss component
            temperature: Temperature scaling for the InfoNCE loss
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.alpha = alpha
        self.temperature = temperature

    def compute_infonce(
        self,
        V_stimulus: torch.Tensor,
        V_physio: torch.Tensor,
        physio_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aligns the Meme Anchor (V_stimulus) to its K valid physiological subjects (V_physio).
        V_stimulus: (B, Hidden_Dim)
        V_physio: (B, Max_Subjects, Hidden_Dim)
        physio_mask: (B, Max_Subjects) -> True for valid subjects
        """
        B, Max_Subj, D = V_physio.shape

        # Normalize vectors for cosine similarity
        V_stim_norm = F.normalize(V_stimulus, p=2, dim=-1)  # (B, D)
        V_phys_norm = F.normalize(V_physio, p=2, dim=-1)  # (B, Max_Subj, D)

        # Calculate similarity between every stimulus and EVERY subject in the batch
        # Reshape physio to (B * Max_Subj, D)
        V_phys_flat = V_phys_norm.view(-1, D)
        mask_flat = physio_mask.view(-1)

        # Only keep valid subjects across the batch
        valid_phys_flat = V_phys_flat[mask_flat]  # (N_valid, D)

        # Sim matrix: (B, N_valid)
        sim_matrix = torch.matmul(V_stim_norm, valid_phys_flat.T) / self.temperature

        # Create target matrix: 1 if subject belongs to that specific meme (anchor), 0 otherwise
        targets = torch.zeros_like(sim_matrix)
        valid_idx_counter = 0
        for b in range(B):
            valid_for_b = physio_mask[b].sum().item()
            targets[b, valid_idx_counter : valid_idx_counter + valid_for_b] = 1.0
            valid_idx_counter += valid_for_b

        # Cross entropy with soft targets (since there are K positives per anchor)
        targets = targets / targets.sum(dim=1, keepdim=True)  # Normalize probabilities
        loss_infonce = -torch.sum(
            targets * F.log_softmax(sim_matrix, dim=1), dim=1
        ).mean()

        return loss_infonce

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
        loss_mtl = (
            (self.lambda_1 * L1.mean())
            + (self.lambda_2 * L2_weighted.sum() / valid_count)
            + (self.lambda_3 * L3_weighted.sum() / valid_count)
        )

        # InfoNCE (Contrastive) Loss
        loss_infonce = self.compute_infonce(
            outputs["V_stimulus"], outputs["V_physio"], masks["physio_mask"]
        )

        # Final Total Loss
        total_loss = loss_mtl + (self.alpha * loss_infonce)

        return {
            "loss_MTL": loss_mtl,
            "loss_InfoNCE": loss_infonce,
            "total_loss": total_loss,
        }
