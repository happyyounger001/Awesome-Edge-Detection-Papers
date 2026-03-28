from build_exe import build_pyinstaller_command


def test_build_command_contains_main_and_config():
    command = build_pyinstaller_command(onefile=False)
    joined = ' '.join(str(part) for part in command)
    assert 'PyInstaller' in joined
    assert 'main.py' in joined
    assert 'config.yaml' in joined


def test_build_command_supports_onefile():
    command = build_pyinstaller_command(onefile=True)
    assert '--onefile' in command
