import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score

from dataset import create_dataloaders
from model_arch.encoder import CNNConfig, CNNEncoder, TransformerConfig, TransformerEncoder
from model_arch.projector import (
    LinearProjectorConfig,
    LinearProjector,
    MLPProjectorConfig,
    MLPProjector,
)

NUM_CLASSES = 6

LLM_CHECKPOINT = "HuggingFaceTB/SmolLM2-360M-Instruct"
# prompt is fixed and identical for every window, so it's tokenized once, not per-example
PROMPT_BEFORE = (
    "Classify the activity as walking, walking upstairs, walking "
    "downstairs, sitting, standing, or laying.\n\nSensor context: "
)
PROMPT_AFTER = "\n\nActivity:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HAR sensor encoder.")
    
    # model
    parser.add_argument("--name", type=str, required=True, help="name of the run")
    parser.add_argument("--model", choices=["direct", "context"], default="direct")
    parser.add_argument("--encoder", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--projector", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--augment", action="store_true", help="apply Gaussian noise augmentation to training windows")

    # optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", choices=["none", "cosine", "step"], default="none")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")

    # training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--eval_metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default = 16)
    parser.add_argument("--num_workers", type=int, default = 0)

    return parser.parse_args()


class DirectClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # [B, 1, embedding_dim]
        x = x.squeeze(1)  # [B, embedding_dim]
        return self.head(x)  # [B, num_classes]


class ContextEmbeddingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        num_classes: int,
        llm_checkpoint: str = LLM_CHECKPOINT,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

        self.tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)
        self.llm = AutoModel.from_pretrained(llm_checkpoint, dtype=torch.float32)  # only backbone, no head needed

        #freeze the parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        llm_dim = self.llm.config.hidden_size
        self.head = nn.Linear(llm_dim, num_classes)

        # tokenize the two fixed halves of the prompt once, split around <SENSOR>
        before_ids = self.tokenizer(PROMPT_BEFORE, add_special_tokens=True, return_tensors="pt").input_ids
        after_ids = self.tokenizer(PROMPT_AFTER, add_special_tokens=False, return_tensors="pt").input_ids
        self.register_buffer("before_ids", before_ids, persistent=False)
        self.register_buffer("after_ids", after_ids, persistent=False)

    def train(self, mode: bool = True):
        # keep the frozen LLM deterministic (no dropout) regardless of the outer model's mode
        super().train(mode)
        self.llm.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))  # [B, 1, llm_dim]

    def forward_from_embedding(self, sensor_embed: torch.Tensor) -> torch.Tensor:
        batch_size = sensor_embed.shape[0]

        embed_tokens = self.llm.get_input_embeddings()
        before_embeds = embed_tokens(self.before_ids).expand(batch_size, -1, -1)  # [B, Lb, llm_dim]
        after_embeds = embed_tokens(self.after_ids).expand(batch_size, -1, -1)  # [B, La, llm_dim]

        inputs_embeds = torch.cat([before_embeds, sensor_embed, after_embeds], dim=1)

        #mask of all ones, because every prompt to LLM has same length, no padding needed
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # hidden state at "Activity:"
        return self.head(last_hidden)  # [B, num_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensor_embed = self.encode(x)  # [B, 1, llm_dim]
        return self.forward_from_embedding(sensor_embed)


def _build_encoder(args: argparse.Namespace) -> tuple[nn.Module, int]:
    if args.encoder == "cnn":
        config = CNNConfig()
        return CNNEncoder(config), config.embedding_dim
    if args.encoder == "transformer":
        config = TransformerConfig()
        return TransformerEncoder(config), config.embedding_dim
    raise ValueError(f"Unknown encoder: {args.encoder}")


def _build_projector(args: argparse.Namespace, model_dim: int) -> nn.Module:
    if args.projector == "linear":
        return LinearProjector(LinearProjectorConfig(model_dim=model_dim))
    if args.projector == "mlp":
        return MLPProjector(MLPProjectorConfig(model_dim=model_dim))
    raise ValueError(f"Unknown projector: {args.projector}")


def build_model(args: argparse.Namespace) -> nn.Module:
    encoder, embedding_dim = _build_encoder(args)

    if args.model == "direct":
        return DirectClassifier(encoder, embedding_dim, num_classes=NUM_CLASSES)

    if args.model == "context":
        projector = _build_projector(args, embedding_dim)
        return ContextEmbeddingModel(encoder, projector, num_classes=NUM_CLASSES)

    raise ValueError(f"Unknown model: {args.model}")

def train_one_epoch(model,train_loader,optimizer,criterion,device):
    model.train()
    total_loss = 0.0
    for inputs,targets in train_loader:
        inputs,targets = inputs.to(device),targets.to(device)

        preds = model(inputs)
        optimizer.zero_grad()

        loss = criterion(preds,targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(train_loader)

def build_optimizer(model, args):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adam":
        return torch.optim.Adam(trainable_params, lr=args.lr)
    if args.optimizer == "sgd":
        return torch.optim.SGD(trainable_params, lr=args.lr)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(optimizer, args):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.lr_scheduler == "step":
        return lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")


def evaluate(model,loader,criterion,device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for input,targets in loader:
            input,targets = input.to(device), targets.to(device)
            preds = model(input)
            total_loss += criterion(preds, targets).item()

            all_preds.append(preds.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds) #combines all batches
    all_targets = torch.cat(all_targets)

    macro_f1 = f1_score(all_targets,all_preds,average="macro")

    return total_loss/len(loader),macro_f1

def visualize(train_loss, val_loss, val_f1, eval_freq, save_root_path):
    save_root_path = Path(save_root_path)
    save_root_path.mkdir(parents=True, exist_ok=True)

    train_epochs = range(1, len(train_loss) + 1)
    val_epochs = [i * eval_freq for i in range(1, len(val_loss) + 1)]

    plt.figure()
    plt.plot(train_epochs, train_loss)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("training loss")
    plt.savefig(save_root_path / "train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(val_epochs, val_loss)
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.title("validation loss")
    plt.savefig(save_root_path / "val_loss.png")
    plt.close()

    plt.figure()
    plt.plot([0, *val_epochs], [0, *val_f1])
    plt.xlabel("epoch")
    plt.ylabel("validation macro-F1")
    plt.title("validation macro-F1")
    plt.ylim(0, 1)
    plt.savefig(save_root_path / "val_f1.png")
    plt.close()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #set seed
    torch.manual_seed(args.seed)

    #load datasets/models/everything else
    train_loader,val_loader,test_loader = create_dataloaders(batch_size = args.batch_size, num_workers = args.num_workers, add_noise = args.augment)
    model = build_model(args).to(device)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1
    best_state = None

    train_losses = []
    val_losses = []
    val_f1s = []

    initial_val_loss, initial_val_f1 = evaluate(model, val_loader, criterion, device)
    print(f"before training: val_loss={initial_val_loss:.4f} val_f1={initial_val_f1:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        if epoch % args.eval_freq == 0:
            val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            #check if best current model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                #store the best model but don't save it to checkpoints yet
                best_state = copy.deepcopy(model.state_dict())

        #updated lr
        if scheduler is not None:
            scheduler.step()

    visualize(train_losses, val_losses, val_f1s, args.eval_freq, Path("graphs") / args.name)

    #now store the best checkpoint, once, after training finishes
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = Path("checkpoints") / f"{args.name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved best model (val_f1={best_val_f1:.4f}) to {checkpoint_path}")

    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()