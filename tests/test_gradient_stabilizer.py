import pytest
import torch

from gradient_stabilizer import GradientStabilizer, GSWrapper


def test_manual_adamw_step_runs():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gs = GradientStabilizer()

    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    gs(optimizer)
    optimizer.step()

    assert torch.isfinite(loss)


def test_wrapper_step_runs():
    model = torch.nn.Linear(4, 2)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = GSWrapper(base_optimizer)

    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_nonfinite_gradient_raises():
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([float("nan"), 1.0])

    gs = GradientStabilizer(raise_on_nonfinite=True)

    with pytest.raises(FloatingPointError):
        gs([p])
