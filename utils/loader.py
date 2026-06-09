import io
import requests
import PIL.Image
import numpy as np

def is_url(path_or_url: str):
    return path_or_url.startswith(('http:', 'https:'))

def imread(path_or_url, max_size=None, mode=None):
    if is_url(path_or_url):
        # wikimedia requires a user agent
        headers = {
            "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(path_or_url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = path_or_url
    img = PIL.Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img) / 255.0
    return img

def load_image(path_or_url, size=128, alpha_premultiply=False):
    if not is_url(path_or_url):
        img = PIL.Image.open(path_or_url)
    else:
        r = requests.get(path_or_url)
        img = PIL.Image.open(io.BytesIO(r.content))
    if img.mode == 'L':
        img = img.convert('RGB')
    if isinstance(size, int):
        size = (size, size)
    img.thumbnail(size, PIL.Image.LANCZOS)
    img = np.float32(img)/255.0
    if alpha_premultiply and img.shape[-1] == 4:
        img[..., :3] *= img[..., 3:]
    elif img.shape[-1] == 3:
        img = np.pad(img, [(0,0),] * (len(img.shape)-1) + [(0,1)], constant_values=1)
    return img

def load_emoji(emoji, **kwargs):
    code_points = [f"{ord(c):04x}" for c in emoji]
    code = '_'.join(code_points)
    url = f'https://github.com/googlefonts/noto-emoji/blob/main/png/512/emoji_u{code}.png?raw=true'
    print(f'{url = }')
    return load_image(url, **kwargs)

def load_target_image(target_str: str, **kwargs):
    if len(target_str) == 1:
        return load_emoji(target_str, **kwargs)
    return load_image(target_str, **kwargs)
