import torch

from coinrun_world_model.models.transformer import ActionConditionedTransformer
from coinrun_world_model.models.vqvae import VQVAE


def test_vqvae_shapes_and_range():
    model = VQVAE(codebook_size=32, code_dim=16, hidden_channels=32)
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out.recon.shape == x.shape
    assert out.indices.shape == (2, 8, 8)
    assert float(out.recon.detach().min()) >= 0.0
    assert float(out.recon.detach().max()) <= 1.0
    decoded = model.decode_indices(out.indices)
    assert decoded.shape == x.shape


def test_transformer_logits_loss_and_action_effect():
    torch.manual_seed(0)
    model = ActionConditionedTransformer(
        codebook_size=32,
        num_actions=15,
        latent_tokens=64,
        layers=1,
        heads=2,
        width=64,
        max_seq_len=256,
    )
    model.eval()
    codes = torch.randint(0, 32, (2, 3, 64))
    actions = torch.tensor([[4, 7], [4, 7]])
    out = model(codes, actions)
    assert out.logits.shape[:2] == out.tokens.shape
    assert out.logits.shape[-1] == 47
    assert out.loss is not None

    context = codes[:, :2]
    left_actions = torch.tensor([[4], [4]])
    right_actions = torch.tensor([7, 7])
    noop_actions = torch.tensor([4, 4])
    with torch.no_grad():
        right = model.generate_next_frame(context, left_actions, right_actions, top_k=1)
        noop = model.generate_next_frame(context, left_actions, noop_actions, top_k=1)
    assert right.shape == (2, 64)
    assert noop.shape == (2, 64)
