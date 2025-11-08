"""
Training Script for Speech Editing Model

Implements:
1. Edit simulation data loader
2. Multi-phase training (token → boundary → full)
3. Logging and checkpointing
4. Evaluation metrics
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
from tqdm import tqdm

from model import SpeechEditModel, SpeechEditConfig
from losses import CombinedEditingLoss
from selective_masking import EditRegion, selective_mask
from mimi_tokenizer import MimiConfig, MimiTokenizer


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Data
    data_dir: str = "./data"
    max_seq_len: int = 2048
    min_edit_frames: int = 5
    max_edit_frames: int = 50
    boundary_frames: int = 4

    # Model
    model_config: SpeechEditConfig = None

    # Training
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    gradient_clip: float = 1.0

    # Loss weights (phase-dependent)
    token_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.1
    speaker_loss_weight: float = 0.05
    duration_loss_weight: float = 0.01

    # Training phases
    phase1_epochs: int = 5  # Token prediction only
    phase2_epochs: int = 5  # + Boundary smoothing
    phase3_epochs: int = 10  # Full objective

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1000
    log_every: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SpeechEditDataset(Dataset):
    """
    Dataset for speech editing

    Simulates edit operations by:
    1. Loading speech token sequences
    2. Randomly selecting edit spans
    3. Creating masked inputs and targets
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 2048,
        min_edit_frames: int = 5,
        max_edit_frames: int = 50,
        boundary_frames: int = 4,
        num_codebooks: int = 8,
    ):
        """
        Args:
            data_dir: Directory with preprocessed token files
            max_seq_len: Maximum sequence length
            min_edit_frames: Minimum edit region size
            max_edit_frames: Maximum edit region size
            boundary_frames: Number of boundary frames
            num_codebooks: Number of codebooks (8 for Mimi)
        """
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.min_edit_frames = min_edit_frames
        self.max_edit_frames = max_edit_frames
        self.boundary_frames = boundary_frames
        self.num_codebooks = num_codebooks

        # Load file list
        # In practice: load .pt files with preprocessed Mimi tokens
        self.file_list = self._load_file_list()

        print(f"Loaded {len(self.file_list)} speech files")

    def _load_file_list(self) -> List[str]:
        """Load list of preprocessed token files"""
        # Placeholder - in practice, scan data_dir for .pt files
        # Each file should contain: {"tokens": [seq_len], "duration": frames}
        file_list = []

        if os.path.exists(self.data_dir):
            for fname in os.listdir(self.data_dir):
                if fname.endswith(".pt"):
                    file_list.append(os.path.join(self.data_dir, fname))

        if len(file_list) == 0:
            print(f"⚠️  No .pt files found in {self.data_dir}")
            print("    Creating synthetic data for testing...")
            # Create synthetic samples
            file_list = ["synthetic"] * 100

        return file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample with simulated edit

        Returns:
            Dictionary with:
            - input_tokens: [seq_len] with masked edit region
            - target_tokens: [seq_len] ground truth
            - edit_mask: [seq_len] boolean mask for edit region
            - boundary_mask: [seq_len] boolean mask for boundaries
            - context_mask: [seq_len] boolean mask for context
        """
        # Load tokens
        if self.file_list[idx] == "synthetic":
            # Create synthetic data for testing
            seq_len = random.randint(400, self.max_seq_len)
            tokens = torch.randint(0, 2048, (seq_len,))
        else:
            # Load from file
            data = torch.load(self.file_list[idx])
            tokens = data["tokens"]

        seq_len = tokens.size(0)

        # Select random edit region
        num_frames = seq_len // self.num_codebooks
        edit_len_frames = random.randint(self.min_edit_frames, self.max_edit_frames)
        edit_len_frames = min(edit_len_frames, num_frames // 2)  # Max 50% of sequence

        edit_start_frame = random.randint(0, num_frames - edit_len_frames)
        edit_end_frame = edit_start_frame + edit_len_frames

        # Create edit region
        edit_region = EditRegion(
            edit_start=edit_start_frame,
            edit_end=edit_end_frame,
            boundary_frames=self.boundary_frames,
            num_codebooks=self.num_codebooks,
        )

        # Get masks
        edit_mask = edit_region.get_edit_mask(seq_len)
        boundary_mask = edit_region.get_boundary_mask(seq_len)
        context_mask = ~edit_mask

        # Create masked input
        mask_ratio = random.uniform(0.2, 0.8)
        input_tokens = selective_mask(
            tokens.unsqueeze(0),
            edit_region,
            mask_ratio=mask_ratio,
            mask_token_id=2048,
        ).squeeze(0)

        return {
            "input_tokens": input_tokens,
            "target_tokens": tokens,
            "edit_mask": edit_mask,
            "boundary_mask": boundary_mask,
            "context_mask": context_mask,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch with padding

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with padding
    """
    # Find max length in batch
    max_len = max(sample["input_tokens"].size(0) for sample in batch)

    # Pad sequences
    batch_dict = {
        "input_tokens": [],
        "target_tokens": [],
        "edit_mask": [],
        "boundary_mask": [],
        "context_mask": [],
        "attention_mask": [],
    }

    for sample in batch:
        seq_len = sample["input_tokens"].size(0)
        pad_len = max_len - seq_len

        # Pad tokens
        input_tokens = F.pad(sample["input_tokens"], (0, pad_len), value=2049)  # PAD
        target_tokens = F.pad(sample["target_tokens"], (0, pad_len), value=2049)

        # Pad masks
        edit_mask = F.pad(sample["edit_mask"].float(), (0, pad_len), value=0.0).bool()
        boundary_mask = F.pad(sample["boundary_mask"].float(), (0, pad_len), value=0.0).bool()
        context_mask = F.pad(sample["context_mask"].float(), (0, pad_len), value=0.0).bool()

        # Attention mask (1 = attend, 0 = ignore padding)
        attention_mask = torch.ones(max_len, dtype=torch.long)
        attention_mask[seq_len:] = 0

        batch_dict["input_tokens"].append(input_tokens)
        batch_dict["target_tokens"].append(target_tokens)
        batch_dict["edit_mask"].append(edit_mask)
        batch_dict["boundary_mask"].append(boundary_mask)
        batch_dict["context_mask"].append(context_mask)
        batch_dict["attention_mask"].append(attention_mask)

    # Stack into batches
    return {k: torch.stack(v) for k, v in batch_dict.items()}


def get_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Get learning rate scheduler with warmup"""

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: CombinedEditingLoss,
    config: TrainingConfig,
    epoch: int,
    phase: str,
) -> Dict[str, float]:
    """
    Train one epoch

    Args:
        model: Speech editing model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: LR scheduler
        loss_fn: Combined loss function
        config: Training config
        epoch: Current epoch
        phase: Training phase ("phase1", "phase2", "phase3")

    Returns:
        Dictionary of average losses
    """
    model.train()
    device = config.device

    # Adjust loss weights based on phase
    if phase == "phase1":
        # Phase 1: Token prediction only
        loss_fn.boundary_loss_weight = 0.0
        loss_fn.speaker_loss_weight = 0.0
    elif phase == "phase2":
        # Phase 2: Add boundary smoothing
        loss_fn.boundary_loss_weight = config.boundary_loss_weight
        loss_fn.speaker_loss_weight = 0.0
    else:  # phase3
        # Phase 3: Full objective
        loss_fn.boundary_loss_weight = config.boundary_loss_weight
        loss_fn.speaker_loss_weight = config.speaker_loss_weight

    total_losses = {"total": 0.0, "token": 0.0, "boundary": 0.0, "speaker": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_tokens = batch["input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        edit_mask = batch["edit_mask"].to(device)
        boundary_mask = batch["boundary_mask"].to(device)
        context_mask = batch["context_mask"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        logits = model(input_tokens, attention_mask)

        # Get embeddings (for boundary/speaker loss)
        # In practice: extract from model's internal activations
        # For now: use dummy embeddings from token embeddings
        generated_embeddings = model.token_embedding(logits.argmax(dim=-1))
        original_embeddings = model.token_embedding(target_tokens)

        # Compute loss
        loss, loss_dict = loss_fn(
            logits=logits,
            target_tokens=target_tokens,
            generated_embeddings=generated_embeddings,
            original_embeddings=original_embeddings,
            edit_mask=edit_mask,
            boundary_mask=boundary_mask,
            context_mask=context_mask,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()
        scheduler.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            total_losses[k] += v
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}",
        })

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses


def main():
    """Main training function"""

    # Create configs
    model_config = SpeechEditConfig(
        vocab_size=2052,
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=2048,
    )

    training_config = TrainingConfig(
        data_dir="./data",
        model_config=model_config,
        batch_size=16,
        num_epochs=20,
        learning_rate=1e-4,
    )

    print("=" * 60)
    print("Speech Editing Training")
    print("=" * 60)
    print(f"Model: {model_config.n_layers} layers, {model_config.d_model} dim")
    print(f"Device: {training_config.device}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print("=" * 60)

    # Create model
    model = SpeechEditModel(model_config).to(training_config.device)
    n_params = model.get_num_params(non_embedding=False)
    print(f"\nModel parameters: {n_params:,} ({n_params / 1e9:.2f}B)")

    # Create dataset and dataloader
    dataset = SpeechEditDataset(
        data_dir=training_config.data_dir,
        max_seq_len=training_config.max_seq_len,
        min_edit_frames=training_config.min_edit_frames,
        max_edit_frames=training_config.max_edit_frames,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for debugging
    )

    # Create loss function
    loss_fn = CombinedEditingLoss(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        boundary_loss_weight=training_config.boundary_loss_weight,
        speaker_loss_weight=training_config.speaker_loss_weight,
    )

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    total_steps = len(dataloader) * training_config.num_epochs
    scheduler = get_lr_schedule(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        total_steps=total_steps,
    )

    # Training loop
    print("\nStarting training...")

    for epoch in range(1, training_config.num_epochs + 1):
        # Determine phase
        if epoch <= training_config.phase1_epochs:
            phase = "phase1"
        elif epoch <= training_config.phase1_epochs + training_config.phase2_epochs:
            phase = "phase2"
        else:
            phase = "phase3"

        # Train epoch
        avg_losses = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            config=training_config,
            epoch=epoch,
            phase=phase,
        )

        # Print losses
        print(f"\nEpoch {epoch} ({phase}) - Losses:")
        for k, v in avg_losses.items():
            print(f"  {k:10s}: {v:.4f}")

        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            os.makedirs(training_config.checkpoint_dir, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": model_config,
                "losses": avg_losses,
            }, checkpoint_path)

            print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
