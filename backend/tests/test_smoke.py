"""Smoke test: the package imports and pytest is wired up."""


def test_core_package_imports():
    import core  # noqa: F401

    assert core is not None
