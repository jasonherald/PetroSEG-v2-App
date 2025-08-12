# Map non-module names used by some code to real modules
import sys

try:
    import skimage.color.colorlabel as _colorlabel
    sys.modules.setdefault("skimage.color.label2rgb", _colorlabel)
except Exception:
    pass

try:
    import skimage.color.colorconv as _colorconv
    # Some code mistakenly treats 'color_dict' like a module; alias to colorconv
    sys.modules.setdefault("skimage.color.color_dict", _colorconv)
except Exception:
    pass

# Also handle io.BytesIO if something tries to import it as a module
try:
    import io as _io
    sys.modules.setdefault("io.BytesIO", _io)
except Exception:
    pass