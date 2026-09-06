import pytest

torch = pytest.importorskip("torch")

from lingbot_recap.visual_token import VisualTokenBottleneck, VisualTokenConfig  # noqa: E402


def test_visual_token_shapes_and_finite_loss():
    config = VisualTokenConfig(
        input_dim=12, token_dim=16, max_tokens=8, heads=4, encoder_layers=1, decoder_layers=1
    )
    model = VisualTokenBottleneck(config)
    embeddings = torch.randn(3, 8, 12)
    valid = torch.ones(3, 8, dtype=torch.bool)
    valid[1, -2:] = False
    output = model.reconstruction_loss(embeddings, valid)
    assert output["loss"].isfinite()
    assert output["token_rms"].isfinite()
    assert model.encode(embeddings, valid).shape == (3, 16)


def test_visual_token_rejects_empty_sample():
    model = VisualTokenBottleneck(
        VisualTokenConfig(input_dim=8, token_dim=8, max_tokens=4, heads=2, encoder_layers=1, decoder_layers=1)
    )
    with pytest.raises(ValueError):
        model.encode(torch.randn(1, 4, 8), torch.zeros(1, 4, dtype=torch.bool))
