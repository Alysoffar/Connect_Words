"""Smoke tests for the LinkWords pipeline package."""


def test_pipeline_package_imports() -> None:
    """Verify that the pipeline package is importable."""
    import linkwords_pipeline

    assert linkwords_pipeline.__doc__
