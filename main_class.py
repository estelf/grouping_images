import argparse
import glob
import re
import time
import os
import cv2
import numpy as np
import umap
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans

import file_operation
import get_Feature
import Partitive_clustering_util


def Feature_list():
    """
    利用できる特徴量一覧を取得する
    """
    with open("get_Feature.py", "r", encoding="utf-8") as f:
        return list(re.findall(r"(?<=def ).*(?=\()", f.read()))

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
        raise Exception(filename+" is faild")

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
            img, 0, 0, padding_half, height-(width+padding_half), cv2.BORDER_CONSTANT, (0, 0, 0))
  # 横長画像→高さを拡張する
    elif width > height:
        padding_img = cv2.copyMakeBorder(
            img, padding_half, width-(height+padding_half), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        padding_img=img
    return cv2.resize(padding_img, (128, 128))


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
                reshape_img.append(self.get_Feature_method(
                    resize_img(my_imread(self.file_list[i]))))
        else:
            # データ読み出し時のアニメーションtqdm風味
            if self.dispvbar:
                bbar = "╪"*(int(idx*50/self.shape[0])-1)+"━"
                wbar = " "*(50-len(bbar))
                print(
                    f"File Access| {idx/self.shape[0]:.3%} |\033[36m{bbar}{wbar}\033[0m| {idx}/{self.shape[0]}", end="\r",flush=True)

            reshape_img = self.get_Feature_method(
                resize_img(my_imread(self.file_list[idx])))

        return reshape_img


# ----コマンドパーサー----
parser = argparse.ArgumentParser(
    description='機械学習を利用した画像分類プログラムです。\n深層学習との連携や各種特徴量を指定できます。')
parser.add_argument("-f", '--folder', help='データのあるフォルダ。', required=True)    #
fea_list = Feature_list()
parser.add_argument("-fe",'--Feature', help=f"特徴量は上記から選んでください。 デフォルト : CLR_UV_YUV",
                    choices=fea_list, default="CLR_UV_YUV")   # よく使う引数なら省略形があると使う時に便利

parser.add_argument('-dr', '--dimension_reduction', help='次元削除を行い処理を高速化します。 小さい数値ほど高速になりますが意図しない結果になる場合もあります。 0で無効 デフォルト : 64',
                    default=64, type=int)   # よく使う引数なら省略形があると使う時に便利
parser.add_argument('-c', '--max_clusters', help='最大クラスタ数。自動が細かすぎる場合値を小さくすると良いです。デフォルト : -1(自動)',
                    default=-1, type=int)   # よく使う引数なら省略形があると使う時に便利

parser.add_argument('-a', '--analyze', default=False,
                    help='シルエット分析による分析結果を出します。処理が重くなります。', action='store_true')   # よく使う引数なら省略形があると使う時に便利
parser.add_argument("-m", "--move", action="store_true",
                    default=False,  help="移動の替わりにコピーにする。")
parser.add_argument("-log", "--logging", action="store_true",
                    default=False,  help="転送ログを保存する")
args = parser.parse_args()


# ----データセット準備----
start = time.perf_counter()  # タイマー開始
DR_flag = args.dimension_reduction  # 次元圧縮フラグ#---*
folder = args.folder

sample_base = folder_to_dataset(
    folder, (Feature_method := eval(f"get_Feature.{args.Feature}")))  # ---*
if DR_flag:
    mapper = umap.UMAP(n_components=DR_flag, random_state=0)
    sample = mapper.fit_transform(sample_base)
    print("\n",flush=True)
else:
    sample = sample_base


print(
    f"| 利用する特徴量\t: {Feature_method.__name__}\t|",flush=True)
abrank = " "
print(
    f"| フォルダ　　　\t: {folder}{abrank*(len(Feature_method.__name__)-len(folder))}\t|",flush=True)
print(
    f"| サンプル数　　\t: {sample.shape[0]}{abrank*(len(Feature_method.__name__)-len(str(sample.shape[0])))}\t|",flush=True)
print(
    f"| データサイズ　\t: {sample.shape[1]}{abrank*(len(Feature_method.__name__)-len(str(sample.shape[1])))}\t|",flush=True)
print(
    f"| 次元削除　　　\t: {DR_flag}{abrank*(len(Feature_method.__name__)-len(str(DR_flag)))}\t|",flush=True)

if args.max_clusters == -1:
    clusters=sample.shape[0]//10
    print(f"| 最大クラスタ　\t: 自動{abrank*(len(Feature_method.__name__)-2)}\t|",flush=True)
else:
    clusters=args.max_clusters
    print(f"| 最大クラスタ　\t: {clusters}{abrank*(len(Feature_method.__name__)-len(str(clusters)))}\t|",flush=True)

if args.analyze :
    print(f"| シルエット分析\t: あり{abrank*(len(Feature_method.__name__)-2)}\t|",flush=True)
else:
    print(f"| シルエット分析\t: なし{abrank*(len(Feature_method.__name__)-2)}\t|",flush=True)
if args.move :
    print(f"| ファイル移動　\t: move{abrank*(len(Feature_method.__name__)-2)}\t|",flush=True)
else:
    print(f"| ファイル移動　\t: copy{abrank*(len(Feature_method.__name__)-2)}\t|",flush=True)

# ----モデル定義----
amount_initial_centers = 10
initial_centers = kmeans_plusplus_initializer(
    sample, amount_initial_centers).initialize()  # ---*
# sample_base.shape[0])#---*

xmeans_instance = xmeans(sample, initial_centers,clusters )


# ----学習実行----
print("---Xmeans計算中---", end="\r",flush=True)
xmeans_instance.process()

# ----学習結果----
clusters = xmeans_instance.get_clusters()
print(f"| 推定クラスタ数\t: {len(clusters)}{abrank*(len(Feature_method.__name__)-len(str(len(clusters))))}\t|",flush=True)
centers = xmeans_instance.get_centers()


# ----使う----
# print(clusters)
labels = Partitive_clustering_util.data_transform.cluster_to_labels(
    sample, clusters)

if args.analyze:
    print("シルエット計算開始")
    sample_base.dispvbar = False  # シルエット分析用にアニメーションを切っておく
    Partitive_clustering_util.Visualization.plot_silhouette(
        sample, labels, f"{Feature_method.__name__}_{len(clusters)}")  # ---*

if args.logging:
    with open(f"{Feature_method.__name__}_{len(clusters)}.txt", "w")as f:
        for i in zip(sample_base.file_list, labels):
            print(f"{i[0]},{i[1]}", file=f)

# ----ファイル移動----
if args.move:
    file_operation.move(folder, sample_base.file_list, labels)  # ---*
else:
    file_operation.copy(folder, sample_base.file_list, labels)  # ---*

print("総実行時間 :", time.perf_counter() - start)
