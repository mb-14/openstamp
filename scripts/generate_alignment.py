#!/usr/bin/env python
"""
Generate semantic alignment transformation matrices.

This script loads pre-computed hidden states and embeddings, then computes
a linear transformation matrix that aligns the hidden states to the embedding
space using either ridge regression with L2 regularization or contrastive
distillation training.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from rich import print as rprint


def ridge_regression(X, Y, alpha):
    """
    Compute ridge regression weights to align X to Y.
    
    Args:
        X: Input features (hidden states) of shape (N, d_in)
        Y: Target embeddings of shape (N, d_out)
        alpha: L2 regularization parameter
    
    Returns:
        W: Transformation matrix of shape (d_in, d_out)
    """
    # L2 normalization of targets
    Y = F.normalize(Y, dim=1)
    d_in = X.shape[1]
    I = torch.eye(d_in, device=X.device)
    # Solve: (X^T X + alpha * I) W = X^T Y
    W = torch.linalg.solve(X.T @ X + alpha * I, X.T @ Y)
    return W


class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


def similarity_matrix(x):
    x = F.normalize(x, dim=-1)
    sim = x @ x.T
    mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    return sim


def distillation_loss(projected, target, temperature=0.07):
    sim_proj = similarity_matrix(projected) / temperature
    sim_tgt = similarity_matrix(target) / temperature

    p = F.log_softmax(sim_proj, dim=-1)
    q = F.softmax(sim_tgt, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


def train_contrastive_alignment(
    hidden_states,
    embeddings,
    device,
    num_epochs=15,
    lr=1e-4,
    batch_size=256,
    temperature=0.07,
    val_split=0.2,
    seed=42,
):
    model = LinearProjector(hidden_states.shape[1], embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = TensorDataset(hidden_states, embeddings)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_proj = model(x)

            loss = distillation_loss(x_proj, y, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_proj = model(x)
                loss = distillation_loss(x_proj, y, temperature=temperature)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        rprint(
            f"[bold cyan]Epoch {epoch + 1}[/bold cyan]: "
            f"Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

    return model


def load_data(dataset_path, embedding_model_name):
    """
    Load hidden states and embeddings from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory
        embedding_model_name: Name of the embedding model (for filename)
    
    Returns:
        hidden_states: Tensor of shape (N, hidden_dim)
        embeddings: Tensor of shape (N, embedding_dim)
    """
    hidden_states_path = os.path.join(dataset_path, "hidden_states.pt")
    embeddings_path = os.path.join(dataset_path, f"embeddings_{embedding_model_name}.pt")
    
    if not os.path.exists(hidden_states_path):
        raise FileNotFoundError(f"Hidden states not found at {hidden_states_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    rprint(f"[bold cyan]Loading hidden states from {hidden_states_path}[/bold cyan]")
    hidden_states = torch.load(hidden_states_path)
    
    rprint(f"[bold cyan]Loading embeddings from {embeddings_path}[/bold cyan]")
    embeddings = torch.load(embeddings_path)
    
    # Ensure same number of samples
    min_samples = min(hidden_states.shape[0], embeddings.shape[0])
    hidden_states = hidden_states[:min_samples]
    embeddings = embeddings[:min_samples]
    
    # Convert to float for computation
    hidden_states = hidden_states.float()
    embeddings = embeddings.float()
    
    rprint(f"[bold green]Loaded {hidden_states.shape[0]} samples[/bold green]")
    rprint(f"  Hidden states shape: {hidden_states.shape}")
    rprint(f"  Embeddings shape: {embeddings.shape}")
    
    return hidden_states, embeddings


def save_alignment_matrix(W, output_path):
    """Save the alignment transformation matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(W, output_path)
    rprint(f"[bold green]Saved alignment matrix to {output_path}[/bold green]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate semantic alignment matrix using ridge regression or contrastive training"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory containing hidden_states.pt and embeddings"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="Name of the embedding model (used to locate embeddings file)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ridge", "contrastive"],
        default="ridge",
        help="Alignment method to use: ridge or contrastive",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="L2 regularization parameter for ridge regression"
    )
    parser.add_argument(
        "--contrastive-epochs",
        type=int,
        default=15,
        help="Number of epochs for contrastive training",
    )
    parser.add_argument(
        "--contrastive-lr",
        type=float,
        default=1e-5,
        help="Learning rate for contrastive training",
    )
    parser.add_argument(
        "--contrastive-batch-size",
        type=int,
        default=256,
        help="Batch size for contrastive training",
    )
    parser.add_argument(
        "--contrastive-temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive distillation loss",
    )
    parser.add_argument(
        "--contrastive-val-split",
        type=float,
        default=0.2,
        help="Validation split ratio for contrastive training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the alignment matrix (defaults to dataset_path)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (default: cpu, recommended for large files)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    rprint(f"[bold yellow]Configuration:[/bold yellow]")
    rprint(f"  Dataset path: {args.dataset_path}")
    rprint(f"  Embedding model: {args.embedding_model}")
    rprint(f"  Method: {args.method}")
    rprint(f"  Alpha (regularization): {args.alpha}")
    rprint(f"  Device: {args.device}")
    rprint(f"  Seed: {args.seed}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Load data
    hidden_states, embeddings = load_data(args.dataset_path, args.embedding_model)
    
    # Move to device if specified
    device = torch.device(args.device)
    if args.device != "cpu":
        rprint(f"[bold yellow]Moving data to {args.device}...[/bold yellow]")
        hidden_states = hidden_states.to(device)
        embeddings = embeddings.to(device)
    
    output_dir = args.output_dir if args.output_dir else args.dataset_path

    if args.method == "ridge":
        # Compute ridge regression alignment
        rprint(
            f"[bold cyan]Computing ridge regression alignment (alpha={args.alpha})...[/bold cyan]"
        )
        W = ridge_regression(hidden_states, embeddings, args.alpha)

        rprint(f"[bold green]Alignment matrix shape: {W.shape}[/bold green]")
        output_path = os.path.join(output_dir, f"align_ridge_{args.embedding_model}.pt")

        # Move back to CPU for saving
        W = W.cpu()
        save_alignment_matrix(W, output_path)
    else:
        rprint(
            "[bold cyan]Training contrastive alignment model "
            f"(epochs={args.contrastive_epochs}, lr={args.contrastive_lr}, "
            f"batch_size={args.contrastive_batch_size}, temperature={args.contrastive_temperature})..."
            "[/bold cyan]"
        )
        model = train_contrastive_alignment(
            hidden_states,
            embeddings,
            device,
            num_epochs=args.contrastive_epochs,
            lr=args.contrastive_lr,
            batch_size=args.contrastive_batch_size,
            temperature=args.contrastive_temperature,
            val_split=args.contrastive_val_split,
            seed=args.seed,
        )

        W = model.proj.weight.data.T
        rprint(f"[bold green]Alignment matrix shape: {W.shape}[/bold green]")
        output_path = os.path.join(output_dir, f"align_contrastive_{args.embedding_model}.pt")

        W = W.cpu()
        save_alignment_matrix(W, output_path)
    
    rprint(f"[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
