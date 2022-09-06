"""
ラベルデータ、ファイルデータをもとにファイル並び変え用モジュール
"""
import os
import shutil

import tqdm

def move(folder,filelist,labels):
    for i in set(labels):
        os.makedirs(f"{folder}{os.sep}{i}",exist_ok=True)
    for i,ii in tqdm.tqdm(zip(filelist,labels), total=len(labels),ascii=" ▖▌▛█",colour="CYAN", desc="ファイル移動"):
        shutil.move(i,f"{folder}{os.sep}{ii}")
def copy(folder,filelist,labels):
    for i in set(labels):
        os.makedirs(f"{folder}{os.sep}{i}",exist_ok=True)
    for i,ii in tqdm.tqdm(zip(filelist,labels), total=len(labels),ascii=" ▖▌▛█",colour="CYAN", desc="ファイルコピー"):
        shutil.copy2(i,f"{folder}{os.sep}{ii}")
