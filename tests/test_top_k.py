import pytest
import torch

from dictionary_learning.trainers.top_k import AutoEncoderTopK


def test_encode_retains_at_most_k_positive_features() -> None:
    torch.manual_seed(0)
    autoencoder = AutoEncoderTopK(activation_dim=4, dict_size=8, k=2)
    encoded = autoencoder.encode(torch.randn(5, 4))
    assert encoded.shape == (5, 8)
    assert torch.all((encoded > 0).sum(dim=-1) <= 2)


def test_checkpoint_round_trip_uses_weights_only_loader(tmp_path) -> None:
    torch.manual_seed(1)
    original = AutoEncoderTopK(activation_dim=4, dict_size=8, k=2)
    checkpoint = tmp_path / "ae.pt"
    torch.save(original.state_dict(), checkpoint)

    loaded = AutoEncoderTopK.from_pretrained(checkpoint, device="cpu")
    inputs = torch.randn(3, 4)
    assert loaded.activation_dim == 4
    assert loaded.dict_size == 8
    assert int(loaded.k.item()) == 2
    assert torch.equal(original(inputs), loaded(inputs))


def test_checkpoint_loader_rejects_wrong_k(tmp_path) -> None:
    checkpoint = tmp_path / "ae.pt"
    torch.save(AutoEncoderTopK(4, 8, 2).state_dict(), checkpoint)
    with pytest.raises(ValueError, match="checkpoint has k=2"):
        AutoEncoderTopK.from_pretrained(checkpoint, k=3)
