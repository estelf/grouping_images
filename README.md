# grouping_images

## 概要

Xmeans法を利用した自動画像分類プログラムです。  
深層学習との連携や各種特徴量を指定できます。  

## 準備

本プログラムはPythonですので実行環境が必要です。  

1. このgitをクローンする
2. クローンしたフォルダ内で`pip install -r req.txt`で依存モジュールをインストールする。(初回のみ)
3. 深層学習との連携機能を使用する場合`Pytorch` `timm`も導入する。[pytorch.org](https://pytorch.org/)

## 使い方

このプログラムはコマンド上で動作します。
対応する画像は、PNG,JPEG,BMP形式です。

```Shell

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        データのあるフォルダ。
  -fe {all_BGR,CLR_all_gray,CLR_H_HLS,CLR_H_HSV,CLR_AB_LAB,CLR_UV_LUV,CLR_UV_YUV,ML_deepL,ML_Laplacian,ML_hog}, --Feature {all_BGR,CLR_all_gray,CLR_H_HLS,CLR_H_HSV,CLR_AB_LAB,CLR_UV_LUV,CLR_UV_YUV,ML_deepL,ML_Laplacian,ML_hog}
                        特徴量は上記から選んでください。 デフォルト : CLR_UV_YUV
  -dr DIMENSION_REDUCTION, --dimension_reduction DIMENSION_REDUCTION
                        次元削除を行い処理を高速化します。 小さい数値ほど高速になりますが意図しない結果になる場合もあります。 0で無効 デフォルト : 64
  -c MAX_CLUSTERS, --max_clusters MAX_CLUSTERS
                        最大クラスタ数。自動が細かすぎる場合値を小さくすると良いです。デフォルト : -1(自動)
  -a, --analyze         シルエット分析による分析結果を出します。処理が重くなります。
  -m, --move            移動の替わりにコピーにする。
  -log, --logging       転送ログを保存する
```  
  
`python .\main_class.py -h` にも同様の説明があります。

## 特徴量

現在10種類の特徴量を選ぶことができます。分類するデータによっていろいろ試してみてください。  
| キー | 説明|備考|
| --- | --- | --- |
| all_BGR | BGR色空間のうち全チャンネルを使う |大まかに分類|
| CLR_all_gray | グレースケールデータを利用する|大まかに分類|
| CLR_H_HLS | HLS色空間のうちHチャンネルを使う|色で大まかに分類|
| CLR_H_HSV | HSV色空間のうちHチャンネルを使う|色で大まかに分類|
| CLR_UV_LUV | LUV色空間のうちU,Vチャンネルを使う|色を重視した分類|
| CLR_UV_YUV | YUV色空間のうちU,Vチャンネルを使う|色を重視した分類|
| CLR_AB_LAB | LAB色空間のうちA,Bチャンネルを使う|色を重視した分類|
| ML_deepL | 学習済みEfficientNet b0を使う|画像の内容に合わせた分類|
| ML_Laplacian | ラプラシアンフィルタを使う|構造を重視した分類|
| ML_hog | HOG特徴量を使う|構造を重視した分類(ベクトル要素あり)|

## シルエット分析

シルエット分析法による分類の可視化とデータの散布図を出力できます。  
計算時間はかかりますが、クラスタリングの指標になります。  
![シルエット](https://github.com/estelf/grouping_images/blob/main/img/CLR_UV_YUV_10.png)
