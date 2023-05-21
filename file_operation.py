"""
ラベルデータ、ファイルデータをもとにファイル並び変え用モジュール
"""
import os
import shutil

import tqdm
import cv2
import glob
import numpy as np
import re


class folder_to_dataset:
    """
    配列アクセスに応じて画像を読み込む
    一部numpyの機能も実装する
    """

    def __init__(self, folder, get_Feature_method) -> None:
        re1 = re.compile(r"png|jpe?g|bmp", re.I)

        self.file_list = [i for i in glob.glob(f"{folder}{os.sep}*.*") if re.search(re1, i)]

        reshape_img = get_Feature_method(my_imread(self.file_list[0]))
        self.shape = len(self.file_list), reshape_img.shape[0]
        self.get_Feature_method = get_Feature_method
        self.dispvbar = True

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if type(idx) is np.ndarray:
            reshape_img = []
            for i in idx:
                reshape_img.append(self.get_Feature_method(resize_img(my_imread(self.file_list[i]))))
        else:
            # データ読み出し時のアニメーションtqdm風味
            if self.dispvbar:
                bbar = "╪" * (int(idx * 50 / self.shape[0]) - 1) + "━"
                wbar = " " * (50 - len(bbar))
                print(
                    f"File Access| {idx/self.shape[0]:.3%} |\033[36m{bbar}{wbar}\033[0m| {idx}/{self.shape[0]}",
                    end="\r",
                    flush=True,
                )

            reshape_img = self.get_Feature_method(resize_img(my_imread(self.file_list[idx])))

        return reshape_img


class count_number:
    def __init__(self):
        self.num = 0

    def step(self):
        self.num = self.num + 1

    def show(self):
        return self.num


def move(folder, filelist, labels):
    for i in set(labels):
        os.makedirs(f"{folder}{os.sep}{i}", exist_ok=True)
    for i, ii in tqdm.tqdm(zip(filelist, labels), total=len(labels), ascii=" ▖▌▛█", colour="CYAN", desc="ファイル移動"):
        shutil.move(i, f"{folder}{os.sep}{ii}")


def copy(folder, filelist, labels):
    for i in set(labels):
        os.makedirs(f"{folder}{os.sep}{i}", exist_ok=True)
    for i, ii in tqdm.tqdm(zip(filelist, labels), total=len(labels), ascii=" ▖▌▛█", colour="CYAN", desc="ファイルコピー"):
        shutil.copy2(i, f"{folder}{os.sep}{ii}")


def my_imread(filename):
    """
    opencv 日本語パス対応
    """
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(e)
        return None


def data_check(folder):
    cont = count_number()

    re1 = re.compile(r"png|jpe?g|bmp", re.I)
    for i in tqdm.tqdm(
        [i for i in glob.glob(f"{folder}{os.sep}*.*") if re.search(re1, i)],
        ascii=" ▖▌▛█",
        colour="CYAN",
        desc="画像エラーチェック",
    ):
        if my_imread(i) is None:
            cont.step()
            os.makedirs(f"{folder}{os.sep}error_file", exist_ok=True)
            shutil.move(i, f"{folder}{os.sep}error_file")
    if cont.show() != 0:
        print(f"{cont.show()}個のエラー画像があります。これらのファイルは除外されます。")


def resize_img(img):
    """
    画像をpaddingしながら128x128にする
    """
    height, width, _ = img.shape  # 画像の縦横サイズを取得
    diffsize = abs(height - width)
    padding_half = int(diffsize / 2)

    # 縦長画像→幅を拡張する
    if height > width:
        padding_img = cv2.copyMakeBorder(
            img, 0, 0, padding_half, height - (width + padding_half), cv2.BORDER_CONSTANT, (0, 0, 0)
        )
    # 横長画像→高さを拡張する
    elif width > height:
        padding_img = cv2.copyMakeBorder(
            img, padding_half, width - (height + padding_half), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
        )
    else:
        padding_img = img
    return cv2.resize(padding_img, (128, 128))
