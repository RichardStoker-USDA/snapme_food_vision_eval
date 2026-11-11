"""Shim for ERNIE-4.5-VL's bundled processor code.

At import it downloads Roboto-Regular.ttf from a Baidu host whose certificate chain fails behind the
cluster proxy, and the engine stops. The font only labels video frames. Redirect that one request to the
same font (Roboto v2.138, Apache-2.0) on GitHub. Loaded only for models with python_shim = "ernie_font".
"""
try:
    import requests as _requests

    _SRC = "paddlenlp.bj.bcebos.com/vision-language-models/materials/Roboto-Regular.ttf"
    _DST = "https://raw.githubusercontent.com/googlefonts/roboto/v2.138/src/hinted/Roboto-Regular.ttf"
    _get = _requests.get

    def _get_redirected(url, *args, **kwargs):
        if isinstance(url, str) and url.endswith(_SRC):
            url = _DST
        return _get(url, *args, **kwargs)

    _requests.get = _get_redirected
except Exception:  # noqa: BLE001 - never block interpreter start
    pass
