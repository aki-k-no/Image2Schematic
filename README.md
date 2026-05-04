# Image to Minecraft 1.8.9 Schematic Workflow

1枚の風景画像から、深度推定と領域分割を使って Minecraft 1.8.9 / FAWE 向けの `.schematic` を生成するワークフローです。

## Pipeline

1. 入力画像をリサイズ
2. `Depth Anything v2` で深度マップを推定
3. `Segment Anything Model (SAM)` でマスク候補を生成
4. マスクごとに色と深度を見て簡易ラベルを推定
5. ラベルと平均色から Minecraft ブロックを選択
6. 深度を `z` 方向に量子化して legacy `.schematic` を出力

## Files

- `src/mcimage2schem/main.py`: CLI エントリポイント
- `src/mcimage2schem/pipeline.py`: メインパイプライン
- `src/mcimage2schem/depth.py`: Depth Anything v2 ラッパー
- `src/mcimage2schem/segment.py`: SAM ラッパー
- `src/mcimage2schem/classify.py`: 領域ラベル推定
- `src/mcimage2schem/blocks.py`: ブロック選択
- `src/mcimage2schem/schematic.py`: Minecraft 1.8.9 向け `.schematic` writer
- `config/workflow.example.json`: 設定例

## Project Folders

- `inputs/`: 変換したい画像を置く場所
- `outputs/`: 生成した `.schematic` の出力先
- `.hf-cache/`: Hugging Face モデルの保存先

## Python Environment

これ以降の Python 作業は `venv` を前提にするのがおすすめです。依存をこのプロジェクトに閉じ込められるので、モデル周りの追加やバージョン調整がかなり楽になります。

PowerShell:

```powershell
& 'C:\Users\okkun\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Requirements

`requirements.txt` には、この workflow を動かすための基本依存をまとめています。

- `numpy`, `Pillow`: 画像処理と配列操作
- `torch`: Depth Anything v2 / SAM の実行基盤
- `transformers`, `accelerate`: Hugging Face モデル実行
- `huggingface-hub`, `safetensors`: モデル取得と安全な重み読み込み
- `scipy`: 前後処理や今後のマスク整形向け
- `tqdm`: 長めの推論処理の進捗表示向け

GPU 版の `torch` を使いたい場合は、環境に合わせて公式配布元から入れ直してください。

## Run

```powershell
. .\.venv\Scripts\Activate.ps1
python -m src.mcimage2schem.main `
  --input .\inputs\scene.png `
  --output .\outputs\scene.schematic `
  --config .\config\workflow.example.json
```

## Notes

- SAM 自体はセマンティック分類をしないため、この実装では `SAM -> 領域切り出し -> 色/深度ベースのヒューリスティック分類` という構成です。
- 生成される構造は「画像正面を保った深度付きレリーフ」です。真上から見た地形変換ではありません。
- 出力は FAWE で扱いやすい legacy `MCEdit .schematic` 形式です。
- ブロック候補は 1.8.9 に存在するものへ制限しています。
- モデル名やブロックパレットは設定ファイルから差し替えられます。

## Next Tuning Ideas

- 風景の種類ごとにブロックパレットを分ける
- SAM のマスクをマージして小領域のノイズを減らす
- CLIP などを追加して領域ラベル推定を学習ベースに寄せる
- `fill_mode=surface` と `solid_to_back` を画像タイプごとに切り替える
