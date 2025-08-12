from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs, copy_metadata

hiddenimports = collect_submodules("sklearn.cluster")
# scikit-learn ships many compiled extensions; include its dylibs
binaries = collect_dynamic_libs("sklearn")
# include metadata (helps versioned plugins)
datas = copy_metadata("scikit-learn")