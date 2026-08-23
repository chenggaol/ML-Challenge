import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import create_dataloaders
from train import build_model, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HAR sensor model on the test set.")

    parser.add_argument("--name", type=str, required=True, help="run name (selects checkpoints/<name>.pt)")
    parser.add_argument("--model", choices=["direct", "context"], default="direct")
    parser.add_argument("--encoder", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--projector", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffled-embedding check")

    return parser.parse_args()


def shuffled_embedding_check(model, loader, criterion, device):
    """Sensor-dependence check: reuse each test example's own real projected
    embedding, but pair it with a different example before the frozen LLM.
    Labels are never touched. No retraining happens here."""
    model.eval()

    all_embeds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            all_embeds.append(model.encode(inputs).cpu())
            all_targets.append(targets)

    all_embeds = torch.cat(all_embeds)  # [N, 1, llm_dim]
    all_targets = torch.cat(all_targets)  # [N]

    shuffled_embeds = all_embeds[torch.randperm(all_embeds.shape[0])]

    total_loss = 0.0
    all_preds = []
    batch_size = loader.batch_size
    num_batches = 0

    with torch.no_grad():
        for start in range(0, len(shuffled_embeds), batch_size):
            batch_embeds = shuffled_embeds[start : start + batch_size].to(device)
            batch_targets = all_targets[start : start + batch_size].to(device)

            logits = model.forward_from_embedding(batch_embeds)
            loss = criterion(logits, batch_targets)

            total_loss += loss.item()
            num_batches += 1
            all_preds.append(logits.argmax(dim=1).cpu())

    all_preds = torch.cat(all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return total_loss / num_batches, macro_f1


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _, _, test_loader = create_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args).to(device)
    checkpoint_path = Path("checkpoints") / f"{args.name}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"[{args.model}] test macro-F1: {test_f1:.4f} (loss={test_loss:.4f})")

    if args.model == "context":
        shuffled_loss, shuffled_f1 = shuffled_embedding_check(model, test_loader, criterion, device)
        print(f"[{args.model}] shuffled-embedding macro-F1: {shuffled_f1:.4f} (loss={shuffled_loss:.4f})")
        print(f"F1 drop after shuffling: {test_f1 - shuffled_f1:.4f}")


if __name__ == "__main__":
    main()
