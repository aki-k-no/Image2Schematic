# Image to Minecraft 1.8.9 Schematic

2D の風景画像から、Minecraft 1.8.9 / FAWE 向けの legacy `.schematic` を生成するプロジェクトです。

今のパイプラインは次の流れです。

1. 入力画像をリサイズ
2. `Depth Anything v2` で相対深度を推定
3. `SAM` で領域マスクを作成
4. 色・深度・位置から領域ラベルを推定
5. カメラ推定と深度から 3D 点群を復元
6. 点群を voxel 化し、補完と厚み付けを適用
7. 色味の近さとラベルを使って 1.8.9 ブロックを選択
8. legacy `MCEdit .schematic` を出力

## Features

- Minecraft 1.8.9 / FAWE 前提の `.schematic` 出力
- `Depth Anything v2` と `SAM` を使った画像解析
- `GeoCalib` 利用のカメラ推定
- デバッグ出力
  - 深度画像
  - SAM オーバーレイ
  - ラベルオーバーレイ
  - block projection
  - camera overlay
  - 3D point cloud PNG / HTML
- CLI から出力 schematic サイズの上書きが可能

## Project Layout

- `src/mcimage2schem/main.py`
  - CLI エントリポイント
- `src/mcimage2schem/pipeline.py`
  - メインパイプライン
- `src/mcimage2schem/depth.py`
  - Depth Anything v2 ラッパー
- `src/mcimage2schem/segment.py`
  - SAM ラッパー
- `src/mcimage2schem/camera.py`
  - カメラ推定と 3D 再投影
- `src/mcimage2schem/classify.py`
  - 領域ラベル推定
- `src/mcimage2schem/blocks.py`
  - ブロック候補と選択ロジック
- `src/mcimage2schem/voxelize.py`
  - voxel 化、補完、厚み付け
- `src/mcimage2schem/schematic.py`
  - legacy `.schematic` writer
- `config/workflow.example.json`
  - 設定例
- `inputs/`
  - 入力画像置き場
- `outputs/`
  - 出力先

## Setup

PowerShell:

```powershell
. .\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`venv` を activate しない場合は、直接 `python.exe` を指定しても動きます。

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

基本実行:

```powershell
python -m src.mcimage2schem.main `
  --input .\inputs\scene.png `
  --output .\outputs\scene.schematic `
  --config .\config\workflow.example.json
```

出力 schematic サイズを CLI で上書き:

```powershell
python -m src.mcimage2schem.main `
  --input .\inputs\scene.png `
  --output .\outputs\scene.schematic `
  --config .\config\workflow.example.json `
  --schem-width 160 `
  --schem-height 96 `
  --schem-length 64
```

## Important Config

`config/workflow.example.json` では主に次を調整できます。

- `image.target_width`, `image.target_height`
  - 入力画像の解析解像度
- `build.target_width`, `build.target_height`, `build.target_length`
  - 出力 schematic の目標サイズ
- `build.forward_distance_scale`
  - 奥行きスケール
- `build.far_distance_boost`, `build.far_distance_power`
  - 後景の距離強調
- `build.shell_enabled`
  - 法線ベースの厚み付け
- `build.fill_column_gaps`
  - 列方向の小さいギャップ埋め
- `build.fill_enclosed_holes`
  - 囲まれた小穴の補完
- `build.connect_neighbors`, `build.fill_triangles`
  - surface 補完

## Debug Output

各実行ごとに `outputs/<出力名>_debug/` が作られます。

主なファイル:

- `01_resized_input.png`
- `02_depth_grayscale.png`
- `03_sam_overlay.png`
- `04_label_overlay.png`
- `05_block_projection.png`
- `06_camera_overlay.png`
- `06b_front_mask_overlay.png`
- `07_camera_space_plot.png`
- `08_camera_space_plot.html`
- `camera.json`
- `forward_distance.json`
- `scale_fit.json`
- `regions.json`
- `points_camera.npy`
- `points_world.npy`

## Current Notes

- `cloud` は `sky` と同様に無視する前提です
- 左右向きは現在の出力側仕様に合わせて固定反転しています
- ブロック選択はラベルだけでなく、各ピクセルの実際の色味も使います
- `.schematic` は 1.8.9 向け legacy 形式です

## Limitations

- 深度は相対深度なので、絶対距離は保証しません
- 単画像からの背面形状は推定であり、完全な 3D 復元ではありません
- 画像によってはラベル分類や距離スケールの調整が必要です
- 高解像度入力は精度向上に効きますが、SAM が重くなります
