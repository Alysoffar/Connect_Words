"""Smoke tests for the LinkWords backend package."""


def test_backend_package_imports() -> None:
    """Verify that the backend package is importable."""
    import linkwords_backend

    assert linkwords_backend.__doc__
