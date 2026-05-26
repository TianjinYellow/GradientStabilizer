import torch
from gradient_stabilizer import GradientStabilizer


def main():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gs = GradientStabilizer(gamma1=0.6, gamma2=0.999)

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()

    gs(optimizer)
    optimizer.step()

    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
