def test_settings_importable():
    from config import settings

    assert settings is not None
