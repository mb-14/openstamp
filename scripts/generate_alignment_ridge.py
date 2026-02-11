#!/usr/bin/env python
"""
Generate semantic alignment transformation matrix using ridge regression.

This script loads pre-computed hidden states and embeddings, then computes
a linear transformation matrix that aligns the hidden states to the embedding
space using ridge regression with L2 regularization.
"""

import os
import argparse
import torch
import torch.nn.functional as F
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
        description="Generate semantic alignment matrix using ridge regression"
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
        "--alpha",
        type=float,
        default=1e-3,
        help="L2 regularization parameter for ridge regression"
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
    rprint(f"  Alpha (regularization): {args.alpha}")
    rprint(f"  Device: {args.device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load data
    hidden_states, embeddings = load_data(args.dataset_path, args.embedding_model)
    
    # Move to device if specified
    device = torch.device(args.device)
    if args.device != "cpu":
        rprint(f"[bold yellow]Moving data to {args.device}...[/bold yellow]")
        hidden_states = hidden_states.to(device)
        embeddings = embeddings.to(device)
    
    # Compute ridge regression alignment
    rprint(f"[bold cyan]Computing ridge regression alignment (alpha={args.alpha})...[/bold cyan]")
    W = ridge_regression(hidden_states, embeddings, args.alpha)
    
    rprint(f"[bold green]Alignment matrix shape: {W.shape}[/bold green]")
    
    # Save the alignment matrix
    output_dir = args.output_dir if args.output_dir else args.dataset_path
    output_path = os.path.join(output_dir, f"align_ridge_{args.embedding_model}.pt")
    
    # Move back to CPU for saving
    W = W.cpu()
    save_alignment_matrix(W, output_path)
    
    rprint(f"[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
