import pandas as pd

from fnews.experiments.image_transfer_baseline import build_model, create_development_split


def test_development_split_is_disjoint_and_stratified() -> None:
    frame = pd.DataFrame(
        {
            "id": [str(index) for index in range(100)],
            "2_way_label": [0] * 60 + [1] * 40,
            "local_path": ["unused"] * 100,
        }
    )

    train, development = create_development_split(frame, 0.2, seed=42)

    assert len(train) == 80
    assert len(development) == 20
    assert set(train["id"]).isdisjoint(development["id"])
    assert development["2_way_label"].mean() == 0.4


def test_resnet_trainable_block_selection() -> None:
    model, label = build_model(pretrained=False, trainable_blocks="layer4")

    assert "trainable=layer4" in label
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_resnet_full_fine_tuning_selection() -> None:
    model, label = build_model(pretrained=False, trainable_blocks="all")

    assert "trainable=all" in label
    assert all(parameter.requires_grad for parameter in model.parameters())
