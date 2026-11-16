from dpow import store


def test_ens_tags_lists_the_whole_family(monkeypatch):
    class R:
        returncode, stdout, stderr = 0, "models\nens-store-2026\nsnowpack\nens-store\nens-store-2025\n", ""

    monkeypatch.setattr(store, "_gh", lambda *a: R)
    assert store.ens_tags() == ["ens-store", "ens-store-2025", "ens-store-2026"]
