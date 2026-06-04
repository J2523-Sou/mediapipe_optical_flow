# MediaPipe Tasks 最小サンプル

`mediapipe.tasks` を使う最小の Hand Landmarker サンプルです。モデルは初回実行時に自動でダウンロードされます。

前提:
- macOS
- Apple Silicon では `mediapipe.tasks` の GPU/Metal 初期化が Python プロセスごと落ちることがあるため、自動で CPU delegate を使います。
- Python 3.9 以上
- カメラを使う場合は macOS のカメラ許可

セットアップ:

```bash
python3 -m pip install -r requirements.txt
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

`realtime_toe_tracker.py` は MediaPipe Pose を毎フレーム実行せず、一定間隔だけ全身検出して、その間はつま先周辺の小さいクロップを OpenCV の光学フローで追跡します。`--detect-every` を大きくすると MediaPipe の実行回数が減り、`--crop-size` で追跡範囲を調整できます。

`openpose_flow_video.py` で動画処理する場合:

```bash
python3 openpose_flow_video.py --video /path/to/video.mov --side left
```

`openpose_flow_video.py` の MediaPipe Tasks 初期化だけ確認したい場合:

```bash
python3 openpose_flow_video.py --check
```

補足:
- 初回は `hand_landmarker.task` を `~/.cache/mediapipe/` に保存します。
- Apple Silicon では `delegate=CPU (Apple Silicon)` と表示されます。
- カメラの取り込みと OpenCV の表示はこの最小構成では CPU です。
- MediaPipe Pose はつま先だけの専用推論にはできないため、クロップは検出後の追跡負荷を下げる目的で使います。
- この venv は Tk 対応の Python で作り直してあり、`tkinter` が使えます。
