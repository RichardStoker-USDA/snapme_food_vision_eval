import tomllib

from snapme_eval import registry
from snapme_eval.gt.snapme import split_for


def test_every_model_file_is_pinned():
    for mid, m in registry.models().items():
        assert m["id"] == mid
        assert len(m["revision"]) == 40, mid
        assert m["engine"] in registry.engines(), mid


def test_engines_pinned_by_digest():
    for name, e in registry.engines().items():
        assert "@sha256:" in e["image"], name


def test_extra_model_dir_wins(tmp_path, monkeypatch):
    m = registry.model("qwen3-5-9b")
    m["hf_repo"] = "example/Override"
    (tmp_path / "qwen3-5-9b.toml").write_text(
        "\n".join(f"{k} = {v!r}".replace("'", '"') for k, v in m.items() if isinstance(v, (str, int, float))))
    monkeypatch.setenv("SNAPME_MODEL_DIRS", str(tmp_path))
    assert registry.model("qwen3-5-9b")["hf_repo"] == "example/Override"
    assert tomllib.loads((tmp_path / "qwen3-5-9b.toml").read_text())


def test_split_is_stable():
    assert split_for("SNAPME101") == split_for("SNAPME101")
    assert {split_for(f"S{i}") for i in range(200)} == {"dev", "test"}
