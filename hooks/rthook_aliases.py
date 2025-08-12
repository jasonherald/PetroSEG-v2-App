import sys
# skimage.segmentation.find_boundaries is a function, not a module
try:
    import skimage.segmentation as _seg
    sys.modules.setdefault("skimage.segmentation.find_boundaries", _seg)
except Exception:
    pass

# sklearn.cluster.KMeans is a class; alias the module name to package
try:
    import sklearn.cluster as _skcl
    sys.modules.setdefault("sklearn.cluster.KMeans", _skcl)
except Exception:
    pass