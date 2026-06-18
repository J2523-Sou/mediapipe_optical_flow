# MediaPipe Tasks CPU サンプル

`mediapipe.tasks` と OpenCV を使って、足先・足部特徴点の検出、追跡、座標CSV化、角速度計算、注釈動画作成を行うスクリプト群です。モデルは初回実行時に自動でダウンロードされます。

前提:
- macOS
- MediaPipe Tasks は CPU delegate 固定
- Python 3.11 系（現在の `.venv` は Python 3.11.15）
- カメラを使う場合は macOS のカメラ許可

プログラム概要:

主に実行するスクリプト:

| ファイル | 概要 |
| --- | --- |
| `toe_mp.py` | 動画の全フレームを MediaPipe Pose で処理し、左足先ランドマークの `x`, `y` 座標を `results/<動画名>_toe_mp.csv` へ保存します。`toe_flow.py` との比較用の基準データに使えます。 |
| `toe_live.py` | カメラまたは動画上で左右どちらかのつま先をリアルタイム追跡します。MediaPipe Pose は一定間隔だけ実行し、間のフレームは小さいクロップ内のオプティカルフローで追跡します。 |
| `toe_flow.py` | 動画ファイルを対象に、つま先座標を MediaPipe Pose と Lucas-Kanade オプティカルフローで追跡します。生座標と Savitzky-Golay フィルター後座標を CSV に出力し、必要なら注釈付き MP4 も作成します。 |
| `trajectory_angle.py` | `frame`, `x`, `y` 系の座標 CSV から、軌跡の進行方向角と角速度を計算して新しい CSV を作成します。`--fps` を指定すると秒単位の角速度も出力します。 |
| `overlay_video.py` | フィルター後座標 CSV を元動画へ重ね、BB 中心まわりの角速度に応じて軌跡の色を変えた MP4 を書き出します。元動画・CSV・BB 中心は GUI または引数で指定できます。 |
| `foot_flow.py` | 足首・踵・つま先から作った足部 ROI 内の複数特徴点を疎なオプティカルフローで追跡し、BB 中心まわりの角速度中央値を `deg/frame`, `deg/s`, `rpm` として出力します。 |

共通モジュール:

| ファイル | 概要 |
| --- | --- |
| `mediapipe_helper.py` | MediaPipe Tasks の CPU delegate 設定、Pose モデルのキャッシュ、Pose Landmarker 初期化をまとめた共通ヘルパーです。 |
| `video_io.py` | OpenCV の MP4 writer 作成と、動画 FPS に基づく MediaPipe 用タイムスタンプ計算を提供する小さな共通ヘルパーです。 |

セットアップ:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

全フレームを MediaPipe Pose で処理してCSVを出力する場合:

```bash
python3 toe_mp.py
```

引数を省略すると `tkinter` のファイル選択ダイアログを使います。
`--video /path/to/video.mov` でも指定できます。出力先は `results/<動画名>_toe_mp.csv` です。

つま先だけをリアルタイム追跡したい場合:

```bash
python3 toe_live.py
```

動画ファイルで試す場合:

```bash
python3 toe_live.py --video /path/to/video.mov --side left
```

`toe_live.py` は MediaPipe Pose を毎フレーム実行せず、一定間隔だけ全身検出して、その間はつま先周辺の小さいクロップを OpenCV のオプティカルフローで追跡します。`--detect-every` を大きくすると MediaPipe の実行回数が減り、`--crop-size` で追跡範囲を調整できます。

`toe_flow.py` で動画処理する場合:

```bash
python3 toe_flow.py --video /path/to/video.mov --side left
```

`toe_mp.py` は全フレームを MediaPipe Pose で検出します。`toe_flow.py` は MediaPipe Pose を一定間隔だけ実行し、その間をオプティカルフローで追跡します。

出力CSVにはフィルタ前の座標 `x`, `y` と、Savitzky-Golayフィルタ後の座標
`x_savgol`, `y_savgol` が格納されます。既定値は窓幅11、多項式次数2です。
変更する場合は `--savgol-window` と `--savgol-polyorder` を指定してください。
窓幅3・多項式次数2のように、多項式次数を窓幅マイナス1にすると入力座標を
そのまま再現して平滑化されないため、指定できません。
検出できなかったフレームは空欄のまま保持し、フィルタは欠損を除いた有効座標列に適用されます。

1点の座標軌跡から、進行方向角とその角速度を計算する場合:

```bash
python3 trajectory_angle.py results/savitky-goley.csv
python3 trajectory_angle.py results/savitky-goley.csv --fps 30
python3 trajectory_angle.py results/savitky-goley.csv --omega-savgol-window 15
```

角速度は軌跡の速度ベクトルが回転する速さです。画像Y軸を反転した直交座標系で、
反時計回りを正とします。足関節などの関節角速度を求めるには、つま先以外の
ランドマークも必要です。通常の数値微分角速度 `omega_*` に加え、アンラップした
進行方向角をSavitzky-Golay微分した `omega_savgol_*` を出力します。角速度用の
既定値は窓幅11・多項式次数2です。

フィルター後座標を元動画へ重ねてMP4を書き出す場合:

```bash
python3 overlay_video.py
```

引数を省略すると、元動画・CSV・表示する窓幅・出力先をGUIで選択できます。
コマンドラインで指定する場合:

```bash
python3 overlay_video.py \
  --video /path/to/IMG_2017.mov \
  --csv results/IMG_2017_toe_flow.csv \
  --bb-frame 0 \
  --angular-speed-threshold 180 \
  --show-raw \
  -o results/IMG_2017_filtered.mp4
```

`savitky-goley.csv` の窓幅11を表示する場合は `--suffix w11` を指定します。
フィルター後座標と直近の軌跡は、進行方向の絶対角速度に応じて緑から黄を経て
赤へ連続的に変わります。`--angular-speed-threshold` は完全な赤になる角速度で、
既定値は180 deg/sです。角速度を計算できない箇所はシアンになります。
`--show-raw` 指定時のフィルター前座標も
赤い十字で表示されます。
角速度は `--bb-x` と `--bb-y` で指定した自転車のBB中心からつま先へ向かう
ベクトルの角度変化として計算します。BB中心は白い斜め十字で表示されます。
`--bb-frame 0` のようにフレーム番号を指定すると、そのフレームを表示して
クリックした位置をBB中心にできます。クリック後にEnterキーで確定します。

足部全体の複数特徴点を疎なオプティカルフローで追跡し、BB周り角速度の中央値を
推定する場合:

```bash
python3 foot_flow.py \
  --video /path/to/IMG_2017.mov \
  --side left \
  --bb-frame 0
```

指定フレーム上でBB中心をクリックしてEnterキーで確定します。MediaPipeで
足首・踵・つま先から足部ROIを定期更新し、ROI内のShi-Tomasi特徴点を
人物セグメンテーションマスクとの交差領域から抽出してLucas-Kanade法で追跡します。
これにより足部の背後にある機材上の特徴点を除外します。前後追跡誤差と角速度のMADで外れ値を除外し、
静止背景に近い点やBB周り角速度がほぼゼロの機材点も除外して、残った点の角速度中央値を
`deg/frame`、`deg/s`、`rpm` としてCSVへ出力します。
注釈動画では追跡点が角速度に応じて緑から赤へ変化します。

`toe_flow.py` の MediaPipe Tasks CPU 初期化だけ確認したい場合:

```bash
python3 toe_flow.py --check
```

補足:
- 初回は `pose_landmarker_full.task` を `~/.cache/mediapipe/` に保存します。
- `--cpu` は古い実行コマンドとの互換用です。現在は指定しなくても常に CPU delegate を使います。
- Apple Silicon では `delegate=CPU (Apple Silicon)` と表示されます。
- カメラの取り込みと OpenCV の表示はこの最小構成では CPU です。
- MediaPipe Pose はつま先だけの専用推論にはできないため、クロップは検出後の追跡負荷を下げる目的で使います。
- この venv は Tk 対応の Python で作り直してあり、`tkinter` が使えます。
