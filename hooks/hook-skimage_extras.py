from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    'io',  # not 'io.BytesIO'
    'skimage.color.color_dict',
    'skimage.color.label2rgb',
    'skimage.segmentation.find_boundaries',
    'sklearn.cluster.KMeans'
]