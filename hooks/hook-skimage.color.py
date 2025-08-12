from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

hiddenimports = collect_submodules("skimage.color")
datas = collect_data_files("skimage") + copy_metadata("scikit-image")