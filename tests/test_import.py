"""Check that the installed package exposes its module boundaries."""

from importlib import import_module


def test_package_imports() -> None:
    for suffix in (
        "",
        ".io",
        ".segmentation",
        ".staging",
    ):
        assert import_module(f"eyespy_spermatogenesis{suffix}") is not None
