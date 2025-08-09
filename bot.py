#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ColorTrendBOT — 12 гармоничных цветов (3×4) с крупными HEX.
После генерации бот спрашивает: «Сохранить как PDF?».

Зависимость: Pillow (PIL). Токен в переменной окружения TELEGRAM_BOT_TOKEN.
"""

from __future__ import annotations

# ---------- stdlib ----------
import colorsys
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from typing import Dict, List, Tuple

# ---------- Pillow ----------
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ======================= Конфигурация токена =======================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TOKEN:
    sys.exit(
        "❗ TELEGRAM_BOT_TOKEN не задан.\n"
        "Пример запуска:\n"
        "  export TELEGRAM_BOT_TOKEN=123456789:AA... && python bot.py"
    )

API_URL = f"https://api.telegram.org/bot{TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{TOKEN}"

# последняя палитра/карточка на чат
LAST_PALETTE: Dict[int, List[Tuple[int, int, int]]] = {}
LAST_CARD_JPEG: Dict[int, bytes] = {}

# =========================== HTTP helpers ==========================

def api_get(method: str, params: dict | None = None) -> dict:
    """GET к Telegram Bot API; вернуть поле 'result'."""
    params = params or {}
    url = f"{API_URL}/{method}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data["result"]

def api_post_multipart(method: str, fields: dict, files: dict) -> dict:
    """
    POST multipart/form-data без внешних библиотек.
    fields: {name: value}; files: {name: (filename, bytes, mimetype)}
    """
    boundary = "------------------------trendpalette"
    parts: List[bytes] = []
    # текстовые поля
    for name, value in fields.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode(), b"\r\n",
        ]
    # файлы
    for name, (filename, blob, mimetype) in files.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            (f'Content-Disposition: form-data; name="{name}"; '
             f'filename="{filename}"\r\n').encode(),
            f"Content-Type: {mimetype}\r\n\r\n".encode(),
            blob, b"\r\n",
        ]
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{API_URL}/{method}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data["result"]

# =========================== Telegram I/O ==========================

def send_message(chat_id: int, text: str) -> None:
    api_get("sendMessage", {"chat_id": chat_id, "text": text})

def send_photo(chat_id: int, image_bytes: bytes, caption: str) -> None:
    files = {"photo": ("palette.jpg", image_bytes, "image/jpeg")}
    fields = {"chat_id": str(chat_id), "caption": caption}
    api_post_multipart("sendPhoto", fields, files)

def send_document(chat_id: int, file_bytes: bytes, filename: str, caption: str = "") -> None:
    files = {"document": (filename, file_bytes, "application/octet-stream")}
    fields = {"chat_id": str(chat_id), "caption": caption}
    api_post_multipart("sendDocument", fields, files)

def get_file_path(file_id: str) -> str:
    return api_get("getFile", {"file_id": file_id})["file_path"]

def download_file(file_path: str) -> bytes:
    with urllib.request.urlopen(f"{FILE_URL}/{file_path}", timeout=60) as resp:
        return resp.read()

# ============================ Цветовые утилиты =====================

def _srgb_to_linear(c: float) -> float:
    c /= 255.0
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    R, G, B = map(_srgb_to_linear, (r, g, b))
    X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
    Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
    Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041
    return X, Y, Z

def xyz_to_lab(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)
    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return xyz_to_lab(*rgb_to_xyz(*rgb))

def delta_e_cie76(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> float:
    return math.sqrt((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2 + (l1[2]-l2[2])**2)

def _is_trendy_rgb(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return (0.20 <= v <= 0.95) and (0.18 <= s <= 0.92)

def extract_trendy_palette(img: Image.Image, want: int = 12) -> List[Tuple[int, int, int]]:
    """Извлечь 12 разнообразных и гармоничных цветов."""
    base = img.convert("RGB")
    base.thumbnail((800, 800), Image.LANCZOS)

    quant = base.quantize(colors=64, method=Image.MEDIANCUT)
    counts = quant.getcolors(maxcolors=quant.width * quant.height) or []
    counts.sort(key=lambda x: x[0], reverse=True)

    pal = quant.getpalette()  # [r,g,b,r,g,b,...]
    def idx_to_rgb(idx: int) -> Tuple[int, int, int]:
        i = idx * 3
        return pal[i], pal[i + 1], pal[i + 2]

    candidates = [idx_to_rgb(idx) for _n, idx in counts]
    candidates = [c for c in candidates if _is_trendy_rgb(c)] or [idx_to_rgb(idx) for _n, idx in counts][:want]

    # объединяем очень близкие тона (по ΔE в Lab)
    merged, labs = [], []
    for c in candidates:
        lab = rgb_to_lab(c)
        if labs and min(delta_e_cie76(lab, l2) for l2 in labs) < 9.0:
            continue
        merged.append(c); labs.append(lab)

    # равномерно по оттенку (hue) в 'want' корзин
    buckets: List[List[Tuple[int, int, int]]] = [[] for _ in range(want)]
    for c in merged:
        h, s, v = colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)
        b = int(h * want) % want
        buckets[b].append(c)

    result: List[Tuple[int, int, int]] = []
    for b in range(want):
        if not buckets[b]:
            continue
        buckets[b].sort(key=lambda c: colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[1], reverse=True)
        result.append(buckets[b][0])
        if len(result) >= want:
            break

    if len(result) < want:
        seen = set(result)
        for c in merged:
            if c not in seen:
                result.append(c); seen.add(c)
                if len(result) >= want:
                    break
    return result[:want]

# ============================ Шрифт ================================

def load_font(size: int) -> ImageFont.ImageFont:
    """Надёжно загружаем TTF (macOS/Win/Linux); иначе fallback на PIL."""
    candidates = [
        "DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

# ========================== Рендер карточки ========================

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    return font.getsize(text)

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def build_swatch_grid_3x4(colors: List[Tuple[int, int, int]]) -> bytes:
    """Сетка 3×4: крупные HEX под каждым свотчем, без нижнего списка."""
    cols, rows = 3, 4
    cell = 340      # размер квадрата цвета
    pad = 40        # внешние поля
    label_h = 120   # высота блока подписи — оптимально
    font_hex = load_font(70)  # КРУПНО, но не гигантски

    W = pad * 2 + cols * cell
    H = pad * 2 + rows * (cell + label_h)

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    text_color = (35, 58, 58)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(colors):
                break
            x = pad + c * cell
            y = pad + r * (cell + label_h)
            draw.rectangle([x, y, x + cell, y + cell], fill=colors[idx])
            hexcode = rgb_to_hex(colors[idx]).lower()
            tw, th = _measure_text(draw, hexcode, font_hex)
            tx = x + (cell - tw) // 2
            ty = y + cell + (label_h - th) // 2
            draw.text((tx, ty), hexcode, fill=text_color, font=font_hex,
                      stroke_width=2, stroke_fill=(255, 255, 255))
            idx += 1

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()

# ====================== Экспорт ZIP и PDF ==========================

def make_palette_zip(hex_list: List[str], filename_base: str = "palette") -> bytes:
    json_bytes = json.dumps({"colors": [{"hex": h} for h in hex_list]},
                            ensure_ascii=False, indent=2).encode("utf-8")
    gpl_lines = ["GIMP Palette", "Name: Trend Palette", "Columns: 3", "#"]
    for h in hex_list:
        r = int(h[1:3], 16); g = int(h[3:5], 16); b = int(h[5:7], 16)
        gpl_lines.append(f"{r:3d} {g:3d} {b:3d}\t{h}")
    gpl_bytes = ("\n".join(gpl_lines) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{filename_base}.json", json_bytes)
        z.writestr(f"{filename_base}.gpl", gpl_bytes)
    return buf.getvalue()

def jpeg_to_pdf_bytes(jpeg_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        if im.mode == "RGBA":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PDF")
        return out.getvalue()

# ====================== Обработка команд/сообщений =================

def handle_text_command(chat_id: int, text: str) -> None:
    t = text.strip()
    low = t.lower()

    if t.startswith("/start"):
        send_message(
            chat_id,
            "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
            "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
        )
        send_message(chat_id, "/pdf — сохранить текущую карточку как PDF")
        return

    if t.startswith("/save"):  # работает, но не рекламируем
        if chat_id not in LAST_PALETTE:
            send_message(chat_id, "Пока нечего сохранять — пришлите фото сначала.")
            return
        hexes = [rgb_to_hex(c) for c in LAST_PALETTE[chat_id]]
        archive = make_palette_zip(hexes, filename_base="palette_12")
        send_document(chat_id, archive, "palette_12.zip", "ZIP: JSON + GPL")
        return

    if low == "/pdf" or low in {"да", "да!", "yes", "y"}:
        if chat_id not in LAST_CARD_JPEG:
            send_message(chat_id, "Нет готовой карточки. Пришлите фото сначала.")
            return
        pdf = jpeg_to_pdf_bytes(LAST_CARD_JPEG[chat_id])
        send_document(chat_id, pdf, "palette_12.pdf", "PDF с палитрой 3×4")
        return

    send_message(chat_id, "Пришлите фотографию или используйте /pdf.")

def handle_update(update: dict) -> None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat_id = msg["chat"]["id"]

    if "text" in msg and isinstance(msg["text"], str):
        handle_text_command(chat_id, msg["text"])
        return

    if "photo" in msg and msg["photo"]:
        largest = max(msg["photo"], key=lambda p: p.get("file_size", 0))
        file_path = get_file_path(largest["file_id"])
        photo = download_file(file_path)
        try:
            with Image.open(io.BytesIO(photo)) as im:
                im = ImageOps.exif_transpose(im)
                palette = extract_trendy_palette(im, want=12)
            LAST_PALETTE[chat_id] = palette
            card = build_swatch_grid_3x4(palette)
            LAST_CARD_JPEG[chat_id] = card
            send_photo(chat_id, card, "")
            send_message(chat_id, "Сохранить как PDF?")
        except Exception as e:
            send_message(chat_id, f"Не удалось обработать фото: {e}")
        return

    send_message(chat_id, "Пришлите фотографию 🙂")

# ================================ Main =============================

def main() -> int:
    offset = None
    print("ColorTrendBOT запущен. Жду сообщения…")
    while True:
        try:
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset
            for upd in api_get("getUpdates", params):
                offset = upd["update_id"] + 1
                handle_update(upd)
        except KeyboardInterrupt:
            print("\nОстановлено пользователем.")
            return 0
        except Exception as e:
            print(f"[warn] {e}", file=sys.stderr)
            time.sleep(2)

if __name__ == "__main__":
    sys.exit(main())
