"""Compatibility shim for Phi-4-reasoning-vision's bundled model code under transformers 5.x.

The model's image processor was written against transformers 4.x, where the SigLIP2 image processor worked
on numpy arrays and its module re-exported helper functions. Transformers 5 moved the helpers and made the
processor torch-only. This shim restores the 4.x behavior the model code relies on: the helper names, numpy
patching and padding, and a numpy base class for rescale and normalize. Loaded only for models whose file
sets python_shim = "siglip2_compat".
"""
try:
    import numpy as _np
    import transformers.image_processing_utils as _ipu
    import transformers.image_transforms as _it
    import transformers.image_utils as _iu
    import transformers.models.siglip2.image_processing_siglip2 as _m
    from transformers import utils as _u

    for _name in ("filter_out_non_signature_kwargs", "ChannelDimension", "to_numpy_array",
                  "infer_channel_dimension_format", "to_channel_dimension_format", "resize",
                  "make_flat_list_of_images", "valid_images", "convert_to_rgb", "validate_preprocess_arguments"):
        if not hasattr(_m, _name):
            for _src in (_u, _iu, _it, _ipu):
                if hasattr(_src, _name):
                    setattr(_m, _name, getattr(_src, _name))
                    break

    def convert_image_to_patches(image, patch_size):
        h, w, c = image.shape
        nh, nw = h // patch_size, w // patch_size
        p = image.reshape(nh, patch_size, nw, patch_size, c).transpose(0, 2, 1, 3, 4)
        return p.reshape(nh * nw, -1)

    def pad_along_first_dim(array, target_length, pad_value=0):
        pad = target_length - array.shape[0]
        mask = _np.ones((target_length,), dtype=_np.int32)
        if pad > 0:
            array = _np.pad(array, [(0, pad)] + [(0, 0)] * (array.ndim - 1), constant_values=pad_value)
            mask[-pad:] = 0
        return array, mask

    class Siglip2ImageProcessor(_ipu.BaseImageProcessor):
        """Numpy SigLIP2 base, as in transformers 4.x, for model code that subclasses it."""

        model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

        def fetch_images(self, images):
            return images

        def rescale(self, image, scale, **kwargs):
            return image.astype(_np.float32) * scale

        def normalize(self, image, mean, std, **kwargs):
            return (image - _np.asarray(mean, _np.float32)) / _np.asarray(std, _np.float32)

    _m.convert_image_to_patches = convert_image_to_patches
    _m.pad_along_first_dim = pad_along_first_dim
    _m.Siglip2ImageProcessor = Siglip2ImageProcessor
except Exception:  # noqa: BLE001 - never block interpreter start
    pass
