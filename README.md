# MediaPipe Tasks CPU サンプル

`mediapipe.tasks` と OpenCV を使って、手・足先・足部特徴点の検出、追跡、座標CSV化、角速度計算、注釈動画作成を行うスクリプト群です。モデルは初回実行時に自動でダウンロードされます。

前提:
- macOS
- MediaPipe Tasks は CPU delegate 固定
- Python 3.11 系（現在の `.venv` は Python 3.11.15）
- カメラを使う場合は macOS のカメラ許可

セットアップ:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

動作確認だけしたい場合:

```bash
python3 mediapipe_gpu_test.py --check
```

カメラで実行する場合:

```bash
python3 mediapipe_gpu_test.py
```

CSV 出力スクリプトは `tkinter` のファイル選択ダイアログを使います。

```bash
python3 mediapipe_csv_xy.py
```

`--video /path/to/video.mov` でも指定できます。

つま先だけをリアルタイム追跡したい場合:

```bash
python3 realtime_toe_tracker.py
```

動画ファイルで試す場合:

```bash
python3 realtime_toe_tracker.py --video /path/to/video.mov --side left
```

`realtime_toe_tracker.py` は MediaPipe Pose を毎フレーム実行せず、一定間隔だけ全身検出して、その間はつま先周辺の小さいクロップを OpenCV のオプティカルフローで追跡します。`--detect-every` を大きくすると MediaPipe の実行回数が減り、`--crop-size` で追跡範囲を調整できます。

`openpose_flow_video.py` で動画処理する場合:

```bash
python3 openpose_flow_video.py --video /path/to/video.mov --side left
```

出力CSVにはフィルタ前の座標 `x`, `y` と、Savitzky-Golayフィルタ後の座標
`x_savgol`, `y_savgol` が格納されます。既定値は窓幅11、多項式次数2です。
変更する場合は `--savgol-window` と `--savgol-polyorder` を指定してください。
窓幅3・多項式次数2のように、多項式次数を窓幅マイナス1にすると入力座標を
そのまま再現して平滑化されないため、指定できません。
検出できなかったフレームは空欄のまま保持し、フィルタは欠損を除いた有効座標列に適用されます。

1点の座標軌跡から、進行方向角とその角速度を計算する場合:

```bash
python3 trajectory_angular_velocity.py results/savitky-goley.csv
python3 trajectory_angular_velocity.py results/savitky-goley.csv --fps 30
python3 trajectory_angular_velocity.py results/savitky-goley.csv --omega-savgol-window 15
```

角速度は軌跡の速度ベクトルが回転する速さです。画像Y軸を反転した直交座標系で、
反時計回りを正とします。足関節などの関節角速度を求めるには、つま先以外の
ランドマークも必要です。通常の数値微分角速度 `omega_*` に加え、アンラップした
進行方向角をSavitzky-Golay微分した `omega_savgol_*` を出力します。角速度用の
既定値は窓幅11・多項式次数2です。

フィルター後座標を元動画へ重ねてMP4を書き出す場合:

```bash
python3 render_filtered_coordinates_video.py
```

引数を省略すると、元動画・CSV・表示する窓幅・出力先をGUIで選択できます。
コマンドラインで指定する場合:

```bash
python3 render_filtered_coordinates_video.py \
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
python3 foot_sparse_flow_angular_velocity.py \
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

`openpose_flow_video.py` の MediaPipe Tasks CPU 初期化だけ確認したい場合:

```bash
python3 openpose_flow_video.py --check
```

補足:
- 初回は `hand_landmarker.task` と `pose_landmarker_full.task` を `~/.cache/mediapipe/` に保存します。
- `--cpu` は古い実行コマンドとの互換用です。現在は指定しなくても常に CPU delegate を使います。
- Apple Silicon では `delegate=CPU (Apple Silicon)` と表示されます。
- カメラの取り込みと OpenCV の表示はこの最小構成では CPU です。
- MediaPipe Pose はつま先だけの専用推論にはできないため、クロップは検出後の追跡負荷を下げる目的で使います。
- この venv は Tk 対応の Python で作り直してあり、`tkinter` が使えます。
