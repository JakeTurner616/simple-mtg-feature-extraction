# mtg_scanner.spec
# -*- mode: python -*-

import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Since we're running PyInstaller from the GUI directory,
# the project root is one level up.
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Collect all submodules from the GUI package.
hiddenimports = collect_submodules('GUI')

a = Analysis(
    [os.path.join(project_root, 'GUI', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[],  # Removed the resources folder from the bundle.
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mtg_scanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mtg_scanner'
)