import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def sobel_lib(input_path, output_path):
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out = np.clip(mag, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, out)

def sobel_native(input_path, output_path):
    img = Image.open(input_path).convert('L')
    w, h = img.size
    pixels = img.load()
    out_img = Image.new('L', (w, h), 0)
    out_pixels = out_img.load()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = (
                - pixels[x - 1, y - 1] + pixels[x + 1, y - 1]
                - 2 * pixels[x - 1, y] + 2 * pixels[x + 1, y]
                - pixels[x - 1, y + 1] + pixels[x + 1, y + 1]
            )
            gy = (
                - pixels[x - 1, y - 1] - 2 * pixels[x, y - 1] - pixels[x + 1, y - 1]
                + pixels[x - 1, y + 1] + 2 * pixels[x, y + 1] + pixels[x + 1, y + 1]
            )
            mag = (gx * gx + gy * gy) ** 0.5
            if mag > 255:
                mag = 255
            out_pixels[x, y] = int(mag)
    out_img.save(output_path)

def run_benchmarks(input_path):
    resolutions = [(200, 150), (400, 300), (800, 600), (1280, 720)]
    lib_times = []
    native_times = []
    labels = []
    base_img = Image.open(input_path).convert('RGB')
    created_files = []
    for w, h in resolutions:
        temp_input = f'temp_in_{w}x{h}.png'
        temp_out_lib = f'temp_lib_{w}x{h}.png'
        temp_out_nat = f'temp_nat_{w}x{h}.png'
        base_img.resize((w, h)).save(temp_input)
        created_files.extend([temp_input, temp_out_lib, temp_out_nat])
        t0 = time.perf_counter()
        sobel_lib(temp_input, temp_out_lib)
        lib_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sobel_native(temp_input, temp_out_nat)
        native_times.append(time.perf_counter() - t0)
        labels.append(f'{w}x{h}')
    for f in created_files:
        if os.path.exists(f):
            os.remove(f)
    plt.figure(figsize=(9, 5))
    x_indices = np.arange(len(resolutions))
    width = 0.35
    plt.bar(x_indices - width / 2, lib_times, width, label='Библиотечная (OpenCV)', color='#2b5c8f')
    plt.bar(x_indices + width / 2, native_times, width, label='Нативная (Python)', color='#e26d5c')
    plt.xticks(x_indices, labels)
    plt.xlabel('Разрешение изображения')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Сравнение быстродействия фильтра Собеля')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark.png', dpi=300)
    plt.close()
    return resolutions, lib_times, native_times

def main():
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
    else:
        input_path = 'input.png'
    t0 = time.perf_counter()
    sobel_lib(input_path, 'output_lib.png')
    t_lib = time.perf_counter() - t0
    t0 = time.perf_counter()
    sobel_native(input_path, 'output_native.png')
    t_native = time.perf_counter() - t0
    print(f'Library execution time: {t_lib:.6f} s')
    print(f'Native execution time: {t_native:.6f} s')
    print(f'Acceleration factor: {t_native / t_lib:.2f}x')
    resolutions, lib_times, native_times = run_benchmarks(input_path)
    for res, lt, nt in zip(resolutions, lib_times, native_times):
        print(f'{res[0]}x{res[1]}: OpenCV = {lt:.5f}s, Native = {nt:.5f}s, Speedup = {nt/lt:.1f}x')

if __name__ == '__main__':
    main()
