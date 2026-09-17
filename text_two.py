# -*- coding: utf-8 -*-
"""
批量读取文件夹中的图片，统一尺寸后输出到新文件夹。

默认策略（改下面参数区即可，不用动代码逻辑）：
    - 目标尺寸     : 800 x 800
    - 适配方式     : 等比缩放 + 居中填充黑边（不裁内容、不变形）
    - 输出位置     : N:\\Opencv_project\\text_2\\picure_one_resized
    - 扫描方式     : 只处理 picure_one 这一层（不递归子文件夹）
    - 输出格式     : 统一 jpg，质量 95，文件名与原图保持一致
    - 额外处理     : 按 EXIF 方向自动旋转、打印日志、同名文件自动加序号

在 VSCode 里直接按 F5 / Ctrl+F5 运行即可，也可以命令行执行：
    python batch_resize.py

依赖：只依赖 opencv-python 和 numpy（numpy 随 opencv-python 一起安装）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ============================ 参数区（按需修改） ============================

# 输入文件夹
INPUT_DIR = r"N:\Opencv_project\text_2\picure_one"

# 输出文件夹（与输入文件夹同级；留空则自动生成 输入文件夹名 + "_resized"）
OUTPUT_DIR = r"N:\Opencv_project\text_2\picure_one_resized"

# 目标尺寸（宽 x 高，单位像素）
TARGET_WIDTH = 800
TARGET_HEIGHT = 800

# 适配方式：
#   "pad"     等比缩放 + 居中填充黑边（推荐，内容不丢、比例不变）
#   "crop"    等比缩放 + 居中裁剪（铺满画面，会切掉边缘）
#   "stretch" 直接拉伸到目标尺寸（会变形）
FIT_MODE = "pad"

# 填充色（BGR 顺序，默认纯黑；例如白色写 (255, 255, 255)）
PAD_COLOR = (0, 0, 0)

# 输出格式：".jpg" 表示统一转成 jpg；None 表示保持原扩展名不变
OUTPUT_EXT = ".jpg"

# JPEG 输出质量（1-100）
JPEG_QUALITY = 95

# 是否递归处理子文件夹
RECURSIVE = False

# 是否按 EXIF 方向自动旋转（手机竖拍照片常用）
AUTO_ROTATE_BY_EXIF = True

# 目标文件已存在时：True = 跳过，False = 自动改名为 xxx_1.jpg
SKIP_EXISTING = False

# 支持的输入格式
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ========================== 参数区结束，下面是逻辑 ==========================


def _setup_console() -> None:
    """输出被重定向到文件/管道时用 UTF-8 写日志，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def read_image(path: Path) -> np.ndarray | None:
    """读取图片，兼容中文/空格路径，并忽略 OpenCV 自带的 EXIF 旋转。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)  # imread 遇到非 ASCII 路径会失败，这里绕开
    except OSError:
        return None
    if buf.size == 0:
        return None
    flags = cv2.IMREAD_COLOR
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= cv2.IMREAD_IGNORE_ORIENTATION  # 统一由本脚本处理 EXIF，避免转两次
    return cv2.imdecode(buf, flags)


def write_image(path: Path, image: np.ndarray) -> bool:
    """保存图片，兼容中文/空格路径。"""
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


# --------------------------- EXIF 方向（纯 Python 解析） ---------------------------


def _parse_tiff_orientation(tiff: bytes) -> int:
    """从 TIFF 数据块里取出 Orientation 标签（0x0112）。"""
    if len(tiff) < 8:
        return 1
    if tiff[:2] == b"II":
        endian = "little"
    elif tiff[:2] == b"MM":
        endian = "big"
    else:
        return 1
    if int.from_bytes(tiff[2:4], endian) != 42:
        return 1

    ifd_offset = int.from_bytes(tiff[4:8], endian)
    if ifd_offset + 2 > len(tiff):
        return 1
    entry_count = int.from_bytes(tiff[ifd_offset:ifd_offset + 2], endian)

    for i in range(entry_count):
        entry = ifd_offset + 2 + i * 12
        if entry + 12 > len(tiff):
            break
        tag = int.from_bytes(tiff[entry:entry + 2], endian)
        if tag == 0x0112:  # Orientation
            value = int.from_bytes(tiff[entry + 8:entry + 10], endian)
            return value if 1 <= value <= 8 else 1
    return 1


def read_exif_orientation(path: Path) -> int:
    """读取 JPEG 的 EXIF 方向标记，失败或不存在时返回 1（正常方向）。"""
    try:
        with open(path, "rb") as f:
            data = f.read(256 * 1024)  # EXIF 一般都在文件头部
    except OSError:
        return 1

    if len(data) < 4 or data[0:2] != b"\xff\xd8":  # 不是 JPEG
        return 1

    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # 到了压缩数据段，不再有 EXIF
            break
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            break
        segment = data[i + 4:i + 2 + seg_len]
        if marker == 0xE1 and segment[:6] == b"Exif\x00\x00":
            return _parse_tiff_orientation(segment[6:])
        i += 2 + seg_len
    return 1


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """按 EXIF 方向标记把图片摆正。"""
    operations = {
        2: lambda im: cv2.flip(im, 1),                                   # 水平镜像
        3: lambda im: cv2.rotate(im, cv2.ROTATE_180),                    # 旋转 180°
        4: lambda im: cv2.flip(im, 0),                                   # 垂直镜像
        5: lambda im: cv2.transpose(im),                                 # 镜像 + 旋转 270°
        6: lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),           # 顺时针 90°
        7: lambda im: cv2.rotate(cv2.transpose(im), cv2.ROTATE_180),     # 镜像 + 旋转 90°
        8: lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),    # 逆时针 90°
    }
    return operations.get(orientation, lambda im: im)(image)


# --------------------------------- 尺寸统一 ---------------------------------


def _pick_interpolation(scale: float) -> int:
    """缩小用 INTER_AREA 更清晰，放大用 INTER_LINEAR。"""
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR


def resize_pad(image: np.ndarray, width: int, height: int, pad_color: tuple) -> np.ndarray:
    """等比缩放后居中放到目标画布上，多出来的部分用 pad_color 填充。"""
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=_pick_interpolation(scale))

    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    left, top = (width - new_w) // 2, (height - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def resize_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """等比缩放后居中裁剪，铺满目标尺寸。"""
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=_pick_interpolation(scale))

    left, top = (new_w - width) // 2, (new_h - height) // 2
    return resized[top:top + height, left:left + width]


def resize_stretch(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """直接拉伸到目标尺寸（宽高比会被改变）。"""
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    return cv2.resize(image, (width, height), interpolation=_pick_interpolation(scale))


def unify_size(image: np.ndarray, width: int, height: int, mode: str, pad_color: tuple) -> np.ndarray:
    """按配置的方式把图片统一成指定尺寸。"""
    mode = mode.lower()
    if mode == "pad":
        return resize_pad(image, width, height, pad_color)
    if mode == "crop":
        return resize_crop(image, width, height)
    if mode == "stretch":
        return resize_stretch(image, width, height)
    raise ValueError(f'不支持的模式：{mode!r}，请使用 "pad" / "crop" / "stretch"')


# --------------------------------- 工具函数 ---------------------------------


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    """收集待处理的图片文件（按名称排序）。"""
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in sorted(input_dir.glob(pattern))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return files


def resolve_output_path(output_dir: Path, source: Path, input_dir: Path, ext: str) -> Path:
    """计算输出路径，必要时保留子目录结构、处理重名。"""
    relative_dir = source.parent.relative_to(input_dir)
    target = output_dir / relative_dir / f"{source.stem}{ext}"

    if SKIP_EXISTING or not target.exists():
        return target

    index = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def main() -> int:
    _setup_console()
    input_dir = Path(INPUT_DIR).expanduser()
    output_dir = Path(OUTPUT_DIR).expanduser() if OUTPUT_DIR else input_dir.with_name(input_dir.name + "_resized")
    ext = OUTPUT_EXT.lower() if OUTPUT_EXT else None

    print("=" * 68)
    print("批量统一图片尺寸")
    print(f"  输入文件夹 : {input_dir}")
    print(f"  输出文件夹 : {output_dir}")
    print(f"  目标尺寸   : {TARGET_WIDTH} x {TARGET_HEIGHT}")
    print(f"  适配方式   : {FIT_MODE}（pad=填充黑边 / crop=裁剪 / stretch=拉伸）")
    print(f"  输出格式   : {ext or '保持原扩展名'}")
    print("=" * 68)

    if not input_dir.is_dir():
        print(f"[错误] 输入文件夹不存在：{input_dir}")
        return 1

    if output_dir.resolve() == input_dir.resolve():
        print("[错误] 输出文件夹不能和输入文件夹相同，否则会覆盖原图。")
        return 1

    files = collect_images(input_dir, RECURSIVE)
    if not files:
        print("[提示] 没有找到可处理的图片，请检查 INPUT_DIR 和 SUPPORTED_EXTS。")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    success, failed, skipped = 0, 0, 0

    for index, source in enumerate(files, start=1):
        tag = f"[{index}/{len(files)}]"
        image = read_image(source)
        if image is None:
            failed += 1
            print(f"{tag} {source.name}  ->  [失败] 无法读取（文件损坏或不是图片）")
            continue

        if AUTO_ROTATE_BY_EXIF:
            orientation = read_exif_orientation(source)
            if orientation != 1:
                image = apply_exif_orientation(image, orientation)

        src_h, src_w = image.shape[:2]
        resized = unify_size(image, TARGET_WIDTH, TARGET_HEIGHT, FIT_MODE, PAD_COLOR)

        target = resolve_output_path(output_dir, source, input_dir, ext or source.suffix.lower())
        if SKIP_EXISTING and target.exists():
            skipped += 1
            print(f"{tag} {source.name}  ->  [跳过] 目标已存在：{target.name}")
            continue

        if not write_image(target, resized):
            failed += 1
            print(f"{tag} {source.name}  ->  [失败] 写入失败：{target}")
            continue

        success += 1
        note = ""
        if FIT_MODE.lower() == "pad":
            scale = min(TARGET_WIDTH / src_w, TARGET_HEIGHT / src_h)
            pad_x = max(0, TARGET_WIDTH - round(src_w * scale))
            pad_y = max(0, TARGET_HEIGHT - round(src_h * scale))
            if pad_x or pad_y:
                note = f"  填充 左右 {pad_x}px / 上下 {pad_y}px"
        print(f"{tag} {source.name}  {src_w}x{src_h} -> {TARGET_WIDTH}x{TARGET_HEIGHT}"
              f"  缩放 {min(TARGET_WIDTH / src_w, TARGET_HEIGHT / src_h):.3f}{note}"
              f"  => {target.relative_to(output_dir)}")

    elapsed = time.perf_counter() - started
    print("-" * 68)
    print(f"处理完成：共 {len(files)} 张，成功 {success}，失败 {failed}，跳过 {skipped}，"
          f"用时 {elapsed:.2f} 秒")
    print(f"输出目录：{output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
