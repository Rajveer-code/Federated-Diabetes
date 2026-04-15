"""
00_config.py
============
DEPRECATION NOTICE
This file is superseded by config_paths.py. Do not edit or add new keys here.
All configuration has been migrated to config_paths.py.
"""
import warnings
warnings.warn(
    '00_config.py is deprecated. Import from config_paths.py instead.',
    DeprecationWarning, stacklevel=2
)
from config_paths import *  # noqa
