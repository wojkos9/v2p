#!/bin/env python3
import av
import numpy as np
import argparse
import os
import numpy as np
from PIL import Image
import sys
from fpdf import FPDF
from skimage.transform import resize

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


def extract_slides(in_file: str, args, out_dir=None, thold=0.01):
    if out_dir is None:
        out_dir = in_file[:in_file.rfind(".")]
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
    last = None
    i = 0
    with av.open(in_file) as container:
        for frame in container.decode(video=0):
            if frame.index % args.step:
                continue
            new = frame.to_ndarray()
            if last is None or mae(last, new) > thold:
                i += 1
                fname = f'{out_dir}/{i:02}.jpeg'
                frame.to_image().save(fname)
                print("Saved", fname)
                last = new
    return out_dir

n = 0

def find_content_bounds(img: np.ndarray, args):
    h, w = img.shape[:2]
    vbounds = [0, w]
    if args.full:
        bounds = [0, h]

    else:
        bounds = [-1, -1]

        col_th = 0.5 if not args.threshold else args.threshold
        window = 10
        min_window = 7
        min_perc = 0.01
        border = int(0.05 * h)
        make_ranges = lambda x: enumerate((range(0, x // 2), range(x-1, x // 2, -1)))
        half_ranges = make_ranges(h)

        avg_black = lambda row: sum(1 if p <= col_th else 0 for p in row) / len(row)

        if args.black:
            for j, ran in half_ranges:
                for i in ran:
                    row = img[i]
                    avg = avg_black(row)
                    if avg < 0.9:
                        bounds[j] = i + 2 * (-1 if j else 1)
                        break
            for j, ran in make_ranges(w):
                for i in ran:
                    col = img[:, i]
                    avg = avg_black(col)
                    if avg < 0.9:
                        vbounds[j] = i + 2 * (-1 if j else 1)
                        break
        else:
            for j, ran in half_ranges:
                stack = []
                val = 0
                for i in ran:
                    row = img[i]
                    m = 1 if avg_black(row) > min_perc else 0

                    stack.append(m)
                    val += m
                    if len(stack) == window:
                        if val >= min_window:
                            bounds[j] = max(0, i - window - border ) if j == 0 else min(h, i + window + border)
                            break
                        val -= stack.pop(0)

    return bounds, vbounds


def montage(in_dir: str, args, per_page=None):
    with cwd(in_dir):
        files = sorted(filter(lambda x: not x.startswith("crop") and not x.endswith("pdf"), os.listdir(".")))
        skip = args.range
        if skip:
            files = files[:skip] if skip < 0 else files[skip:]
        pdf = FPDF()
        pdf.add_page()
        offset = 0
        for i, (img, f) in enumerate(((np.asfarray(Image.open(f).convert('L')) / 255, f) for f in files)):
            new_f = "crop_"+f
            sys.stderr.write(f"{i}/{len(files)} ({f}): ")
            width = img.shape[1]
            (a, b), (c, d) = find_content_bounds(img, args=args)
            if a == -1 or b == -1 or d <= c:
                print("Fail:", a, b)
                continue

            crop: np.ndarray = img[a:b, c:d]
            sc = width/(d-c)
            frag_h = int((b - a) * sc)
            crop = resize(crop, (frag_h, width))

            delta = frag_h * pdf.w / width

            if per_page:
                if i and i % per_page == 0:
                    pdf.add_page()
            elif offset + delta >= pdf.h:
                    offset = 0
                    pdf.add_page()

            crop = (np.uint8(crop * 255)).repeat(3, -1).reshape((frag_h, width, 3))
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
    par.add_argument("-b", dest="black", action="store_true")
    par.add_argument("-a", dest="all", action="store_true")
    par.add_argument("-f", dest="full", action="store_true")
    par.add_argument("-r", dest="range", type=int)
    par.add_argument("-s", dest="step", type=int, default=100)
    par.add_argument("-t", dest="threshold", type=float)
    args = par.parse_args()

    dir = args.in_dir
    if args.in_file:
        dir = extract_slides(args.in_file, args, args.out_dir)
    if dir or args.all:
        montage(dir, args)