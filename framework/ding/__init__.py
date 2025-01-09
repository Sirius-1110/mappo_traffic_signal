import os

__TITLE__ = 'Framework'
__VERSION__ = 'v0.4.8'
__DESCRIPTION__ = 'MARL AI Engine'
__AUTHOR__ = "Contributors"
__AUTHOR_EMAIL__ = "Flow"
__version__ = __VERSION__

enable_hpc_rl = os.environ.get('ENABLE_DI_HPC', 'false').lower() == 'true'
enable_linklink = os.environ.get('ENABLE_LINKLINK', 'false').lower() == 'true'
enable_numba = True
