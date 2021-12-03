#!/bin/env python3
import av
import numpy as np
import argparse
import os
import numpy as np
from PIL import Image
import sys
from fpdf import FPDF

from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def mae(a, b):
    mae = np.sum(np.absolute((b.astype("float") - a.astype("float"))))
    mae /= float(a.shape[0] * a.shape[1] * 255)
    return mae


def extract_slides(in_file: str, out_dir=None, thold=0.01, step=100):
    if out_dir is None:
        out_dir = in_file[:in_file.rfind(".")]
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
    last = None
    i = 0
    with av.open(in_file) as container:
        for frame in container.decode(video=0):
            if frame.index % step:
                continue
            new = frame.to_ndarray()
            if last is None or mae(last, new) > thold:
                i += 1
                frame.to_image().save(f'{out_dir}/{i:02}.jpeg')
                last = new

def find_content_bounds(img: np.ndarray):
    h, w = img.shape[:2]

    bounds = [-1, -1]

    start_ix = 10
    col_th = 0.2
    window = 10
    min_window = 7
    min_perc = 0.01
    border = int(0.05 * h)

    for j, r in enumerate((range(start_ix+1, h // 2), range(h-1, h // 2, -1))):
        stack = []
        val = 0
        for i in r:
            row = img[i]
            m = 1 if sum(1 if p <= col_th else 0 for p in row) > min_perc * w else 0

            stack.append(m)
            val += m
            if len(stack) == window:
                if val >= min_window:
                    bounds[j] = max(0, i - window - border ) if j == 0 else min(h, i + window + border)
                    break
                val -= stack.pop(0)

    return bounds, [0, w]


def montage(in_dir: str, n=-1, per_page=None):
    with cwd(in_dir):
        files = sorted(filter(lambda x: not x.startswith("crop"), os.listdir(".")))[:n]
        pdf = FPDF()
        pdf.add_page()
        offset = 0
        for i, (img, f) in enumerate(((np.asfarray(Image.open(f).convert('L')) / 255, f) for f in files)):
            new_f = "crop_"+f
            sys.stderr.write(f"{i}/{len(files)} ({f}): ")
            (a, b), (_, w) = find_content_bounds(img)
            if a == -1 or b == -1:
                print("Fail:", a, b)
                return

            frag_h = b - a
            delta = frag_h * pdf.w / w

            if per_page:
                if i and i % per_page == 0:
                    pdf.add_page()
            elif offset + delta >= pdf.h:
                    offset = 0
                    pdf.add_page()
            crop: np.ndarray = np.uint8(img[a:b] * 255)
            crop = crop.repeat(3, -1).reshape((frag_h, w, 3))
            Image.fromarray(crop).save(new_f)

            pdf.image(new_f, 0, offset, w=pdf.w)
            offset += delta

            sys.stderr.write(str((a,b))+"\n")
        pdf.output("out.pdf")

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("-i", dest="in_file", type=str)
    par.add_argument("-d", dest="in_dir", type=str)
    par.add_argument("-o", dest="out_dir", type=str)
    args = par.parse_args()

    if args.in_file:
        extract_slides(args.in_file, args.out_dir)
    elif args.in_dir:
        montage(args.in_dir, -3)