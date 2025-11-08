"""
Selective Masking Strategy for Speech Editing

This module implements the core innovation: selective remasking that targets
only the edit region while preserving full bidirectional context.

Key differences from standard masked diffusion:
1. Only edit region is masked (context always visible)
2. Boundary-aware unmasking (prioritizes boundaries first)
3. Region-specific attention (different handling for edit vs context)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class EditRegion:
    """
    Defines an edit region with boundaries

    Attributes:
        edit_start: Starting frame index of edit
        edit_end: Ending frame index of edit (exclusive)
        boundary_frames: Number of boundary frames for prosody smoothing
        num_codebooks: Number of codebooks in RVQ (default 8 for Mimi)
    """
    edit_start: int
    edit_end: int
    boundary_frames: int = 4
    num_codebooks: int = 8

    def to_token_indices(self) -> Tuple[int, int]:
        """Convert frame indices to token indices (with delay pattern)"""
        # With delay pattern, each frame has num_codebooks tokens
        token_start = self.edit_start * self.num_codebooks
        token_end = self.edit_end * self.num_codebooks
        return token_start, token_end

    def get_edit_mask(self, seq_len: int) -> torch.Tensor:
        """
        Get boolean mask for edit region

        Args:
            seq_len: Total sequence length in tokens

        Returns:
            edit_mask: [seq_len] boolean tensor (True for edit region)
        """
        token_start, token_end = self.to_token_indices()
        edit_mask = torch.zeros(seq_len, dtype=torch.bool)
        edit_mask[token_start:token_end] = True
        return edit_mask

    def get_boundary_mask(self, seq_len: int) -> torch.Tensor:
        """
        Get boolean mask for boundary regions

        Args:
            seq_len: Total sequence length in tokens

        Returns:
            boundary_mask: [seq_len] boolean tensor (True for boundaries)
        """
        token_start, token_end = self.to_token_indices()

        # Left boundary: [edit_start - boundary_frames, edit_start)
        left_start = max(0, (self.edit_start - self.boundary_frames) * self.num_codebooks)
        left_end = token_start

        # Right boundary: [edit_end, edit_end + boundary_frames)
        right_start = token_end
        right_end = min(seq_len, (self.edit_end + self.boundary_frames) * self.num_codebooks)

        boundary_mask = torch.zeros(seq_len, dtype=torch.bool)
        boundary_mask[left_start:left_end] = True
        boundary_mask[right_start:right_end] = True

        return boundary_mask


def selective_mask(
    tokens: torch.Tensor,
    edit_region: EditRegion,
    mask_ratio: float,
    mask_token_id: int = 2048,
) -> torch.Tensor:
    """
    Selectively mask only the edit region

    Key innovation: Context region is NEVER masked, only edit region is masked
    according to mask_ratio.

    Args:
        tokens: [batch_size, seq_len] input token sequence
        edit_region: EditRegion defining what to mask
        mask_ratio: Fraction of edit region to mask (0.0 to 1.0)
        mask_token_id: Token ID for [MASK] token

    Returns:
        masked_tokens: [batch_size, seq_len] with edit region partially masked
    """
    batch_size, seq_len = tokens.shape
    masked_tokens = tokens.clone()

    # Get edit mask
    edit_mask = edit_region.get_edit_mask(seq_len).to(tokens.device)

    # Expand to batch dimension
    edit_mask = edit_mask.unsqueeze(0).expand(batch_size, -1)

    # Create random mask within edit region
    random_mask = torch.rand_like(tokens, dtype=torch.float) < mask_ratio

    # Apply mask only to edit region
    mask_positions = edit_mask & random_mask
    masked_tokens[mask_positions] = mask_token_id

    return masked_tokens


def iterative_unmask_with_boundaries(
    model,
    masked_tokens: torch.Tensor,
    edit_region: EditRegion,
    num_steps: int = 12,
    mask_token_id: int = 2048,
    temperature: float = 1.0,
    boundary_boost: float = 0.2,
) -> torch.Tensor:
    """
    Iteratively unmask edit region with boundary-aware scheduling

    Key innovation: Unmask from boundaries inward for prosody continuity

    Args:
        model: Speech editing model
        masked_tokens: [batch_size, seq_len] initially masked sequence
        edit_region: EditRegion defining edit and boundary regions
        num_steps: Number of iterative unmasking steps
        mask_token_id: Token ID for [MASK] token
        temperature: Sampling temperature (1.0 = deterministic argmax)
        boundary_boost: Confidence boost for boundary tokens (unmask earlier)

    Returns:
        unmasked_tokens: [batch_size, seq_len] fully unmasked sequence
    """
    batch_size, seq_len = masked_tokens.shape
    device = masked_tokens.device

    current_tokens = masked_tokens.clone()

    # Get masks
    edit_mask = edit_region.get_edit_mask(seq_len).to(device)
    boundary_mask = edit_region.get_boundary_mask(seq_len).to(device)

    # Expand to batch
    edit_mask = edit_mask.unsqueeze(0).expand(batch_size, -1)
    boundary_mask = boundary_mask.unsqueeze(0).expand(batch_size, -1)

    for step in range(num_steps):
        # Forward pass
        with torch.no_grad():
            logits = model(current_tokens)  # [batch, seq_len, vocab_size]

        # Get confidence scores (max probability)
        probs = F.softmax(logits / temperature, dim=-1)
        confidence = probs.max(dim=-1).values  # [batch, seq_len]

        # Unmask schedule: linear from 100% masked to 0% masked
        target_mask_ratio = 1.0 - (step + 1) / num_steps

        # Boundary boost: unmask boundaries earlier (first 1/3 of steps)
        if step < num_steps // 3:
            confidence = confidence.clone()  # Avoid in-place modification
            confidence[boundary_mask] += boundary_boost

        # Identify currently masked positions in edit region
        masked_positions = (current_tokens == mask_token_id) & edit_mask
        num_masked = masked_positions.sum(dim=1, keepdim=True)  # [batch, 1]

        # Calculate how many to unmask
        num_to_unmask = (num_masked * (1 - target_mask_ratio)).long()
        num_to_unmask = torch.clamp(num_to_unmask, min=1)  # Unmask at least 1

        # Select tokens to unmask based on confidence
        for b in range(batch_size):
            if num_masked[b] == 0:
                continue

            # Get confidence for masked positions in this batch
            masked_conf = confidence[b].clone()
            masked_conf[~masked_positions[b]] = -float('inf')

            # Select top-k most confident
            k = min(num_to_unmask[b].item(), masked_positions[b].sum().item())
            if k == 0:
                continue

            topk_vals, topk_indices = torch.topk(masked_conf, k)

            # Unmask selected positions
            predicted_tokens = logits[b].argmax(dim=-1)
            current_tokens[b, topk_indices] = predicted_tokens[topk_indices]

    return current_tokens


def create_edit_region_from_frames(
    edit_start_frame: int,
    edit_end_frame: int,
    boundary_frames: int = 4,
    num_codebooks: int = 8,
) -> EditRegion:
    """
    Convenience function to create EditRegion from frame indices

    Args:
        edit_start_frame: Starting frame index
        edit_end_frame: Ending frame index (exclusive)
        boundary_frames: Number of boundary frames for smoothing
        num_codebooks: Number of codebooks in RVQ

    Returns:
        EditRegion object
    """
    return EditRegion(
        edit_start=edit_start_frame,
        edit_end=edit_end_frame,
        boundary_frames=boundary_frames,
        num_codebooks=num_codebooks,
    )


def create_edit_region_from_time(
    edit_start_sec: float,
    edit_end_sec: float,
    frame_rate: float = 12.5,
    boundary_frames: int = 4,
    num_codebooks: int = 8,
) -> EditRegion:
    """
    Convenience function to create EditRegion from time in seconds

    Args:
        edit_start_sec: Starting time in seconds
        edit_end_sec: Ending time in seconds
        frame_rate: Frame rate in Hz (12.5 Hz for Mimi)
        boundary_frames: Number of boundary frames for smoothing
        num_codebooks: Number of codebooks in RVQ

    Returns:
        EditRegion object
    """
    edit_start_frame = int(edit_start_sec * frame_rate)
    edit_end_frame = int(edit_end_sec * frame_rate)

    return EditRegion(
        edit_start=edit_start_frame,
        edit_end=edit_end_frame,
        boundary_frames=boundary_frames,
        num_codebooks=num_codebooks,
    )


# Example usage
if __name__ == "__main__":
    # Example: Edit 2 seconds of speech from 1.0s to 3.0s
    edit_region = create_edit_region_from_time(
        edit_start_sec=1.0,
        edit_end_sec=3.0,
        frame_rate=12.5,  # Mimi frame rate
    )

    print(f"Edit region: frames {edit_region.edit_start} to {edit_region.edit_end}")
    print(f"Token indices: {edit_region.to_token_indices()}")

    # Create dummy tokens
    seq_len = 400  # 50 frames * 8 codebooks
    tokens = torch.randint(0, 2048, (2, seq_len))  # batch_size=2

    # Selective masking
    masked_tokens = selective_mask(
        tokens,
        edit_region,
        mask_ratio=0.7,
        mask_token_id=2048,
    )

    # Check masking
    edit_mask = edit_region.get_edit_mask(seq_len)
    num_masked_in_edit = (masked_tokens[0, edit_mask] == 2048).sum().item()
    num_masked_in_context = (masked_tokens[0, ~edit_mask] == 2048).sum().item()

    print(f"\nMasked tokens in edit region: {num_masked_in_edit}")
    print(f"Masked tokens in context: {num_masked_in_context}")
    print(f"✓ Context preservation: {num_masked_in_context == 0}")
