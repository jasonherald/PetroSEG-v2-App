from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

collect_submodules('scipy.special')
collect_dynamic_libs('scipy')