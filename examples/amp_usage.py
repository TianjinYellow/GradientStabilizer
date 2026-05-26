import torch
from gradient_stabilizer import GradientStabilizer


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available; AMP example is intended for CUDA.")
        return

    device = torch.device("cuda")
    model = torch.nn.Linear(10, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gs = GradientStabilizer(gamma1=0.6, gamma2=0.999)
    scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(32, 10, device=device)
    y = torch.randn(32, 1, device=device)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        loss = torch.nn.functional.mse_loss(model(x), y)

    scaler.scale(loss).backward()

    # Important: GradientStabilizer should see true, unscaled gradients.
    scaler.unscale_(optimizer)
    gs(optimizer)

    scaler.step(optimizer)
    scaler.update()

    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
