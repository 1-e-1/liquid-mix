import inspect

import torch

import liquid_mix.pure_fluid_vlm as pure
from liquid_mix.pure_fluid_vlm import PureFluidConfig, PureFluidVLM, SimpleCaptionTokenizer


def test_caption_tokenizer_is_deterministic_and_pads():
    tokenizer = SimpleCaptionTokenizer.fit(
        ["Red dragon with wings", "blue dragon on water"],
        max_vocab=8,
        max_len=6,
    )

    ids = tokenizer.encode("red unknown-token dragon")

    assert ids.shape == (6,)
    assert ids[0].item() == tokenizer.token_to_id["red"]
    assert ids[1].item() == 1
    assert ids[-1].item() == 0
    restored = SimpleCaptionTokenizer.from_state_dict(tokenizer.state_dict())
    assert torch.equal(ids, restored.encode("red unknown-token dragon"))


def test_pure_fluid_vlm_outputs_pair_scores_and_score_matrix():
    config = PureFluidConfig(vocab_size=24, max_len=8, hidden_dim=32, field_grid=2, text_layers=1, text_heads=4)
    model = PureFluidVLM(config)
    images = torch.randn(3, 3, 64, 64)
    tokens = torch.randint(0, config.vocab_size, (3, config.max_len))
    tokens[:, -1] = 0

    pair_scores = model(images, tokens)
    score_matrix = model.score_matrix(images, tokens, chunk_size=4)

    assert pair_scores.shape == (3,)
    assert score_matrix.shape == (3, 3)
    assert torch.isfinite(pair_scores).all()
    assert torch.isfinite(score_matrix).all()


def test_pure_fluid_module_has_no_transformers_dependency():
    source = inspect.getsource(pure)

    assert "transformers" not in source
    assert "CLIPModel" not in source
    assert "CLIPProcessor" not in source
