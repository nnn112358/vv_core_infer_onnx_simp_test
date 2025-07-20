# VV Core Inference Simple

## このプロジェクトについて

### Fork元
https://github.com/Hiroshiba/vv_core_inference


VV Core Inference SimpleはHiroshiba氏の「vv_core_inference」をベースに、AXERAのNPUで動作することを目指し、簡潔にしたプロジェクトです。音声合成を行うために、以下の4つのモデルを組み合わせて使用します。


### 使用するモデル

- **yukarin_s**: テキストから音の長さ（継続時間）を予測
- **yukarin_sa**: 音の高低（イントネーション）を予測  
- **yukarin_sosoa**: 音の特徴（スペクトログラム）を生成
- **HiFi-GAN**: 高品質な音声ファイルを合成するボコーダ

## 必要な環境

- Python 3.11以上
- UVパッケージマネージャー

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. モデルファイルの準備

ONNXモデルファイルを取得するには、以下のコマンドを実行してください：

```bash
git clone https://github.com/Hiroshiba/vv_core_inference
uv run convert.py \
  --yukarin_s_model_dir "model/yukarin_s" \
  --yukarin_sa_model_dir "model/yukarin_sa" \
  --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
  --hifigan_model_dir "model/hifigan"
```

## 使い方

### 基本的な使用例

```bash
uv run run.py \
  --yukarin_s_model_dir "model/yukarin_s" \
  --yukarin_sa_model_dir "model/yukarin_sa" \
  --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
  --hifigan_model_dir "model/hifigan" \
  --speaker_ids 5 \
  --texts "おはようございます、こんにちは、こんばんは、どうでしょうか"
```

### シンプルな実行

CPUのみで動作するため、追加のセットアップは不要です：

```bash
uv run run.py --texts "こんにちは" --speaker_ids 5
```

### コマンドオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--yukarin_s_model_dir` | yukarin_sモデルのフォルダパス | `model/yukarin_s` |
| `--yukarin_sa_model_dir` | yukarin_saモデルのフォルダパス | `model/yukarin_sa` |
| `--yukarin_sosoa_model_dir` | yukarin_sosoaモデルのフォルダパス | `model/yukarin_sosoa` |
| `--hifigan_model_dir` | HiFi-GANモデルのフォルダパス | `model/hifigan` |
| `--use_gpu` | GPU加速を有効にする（このプロジェクトでは未使用） | なし |
| `--texts` | 音声合成したいテキスト（複数指定可能） | 必須 |
| `--speaker_ids` | 使用する話者ID（複数指定可能） | 必須 |
| `--method` | 推論方法（現在は`onnx`のみサポート） | `onnx` |

## 出力ファイル

生成された音声ファイルは、以下の形式でカレントディレクトリに保存されます：

```
onnx-{テキスト内容}-{話者ID}.wav
```

**音声の仕様**
- サンプルレート: 24kHz
- 形式: WAVファイル

## プロジェクトの構成

```
├── model/                     # 音声合成モデル
│   ├── hifigan/              # 音声生成用ボコーダ
│   ├── yukarin_s/            # 音の長さ予測
│   ├── yukarin_sa/           # 音の高低予測
│   └── yukarin_sosoa/        # 音の特徴生成
├── vv_core_inference/        # コア機能
│   ├── forwarder.py          # メイン処理
│   ├── onnx_*.py            # ONNX用の処理
│   └── utility.py           # 補助機能
├── run.py                    # 実行用スクリプト
└── pyproject.toml           # プロジェクト設定
```

## 使用ライブラリ

このプロジェクトは以下のライブラリを使用しています：

- **onnxruntime**: ONNXモデルでの推論実行
- **soundfile**: 音声ファイルの読み書き
- **numpy**: 数値計算
- **torch**: PyTorchフレームワーク
- **各種VOICEVOXモデル**: 音声合成の各段階で使用

## ライセンス

各モデルのライセンスについては、それぞれのモデルリポジトリを参照してください。

## トラブルシューティング

### よくある問題

**モデルファイルが見つからない場合**
- モデルディレクトリのパスが正しいか確認
- ONNXモデルの変換が完了しているか確認

**音声ファイルが生成されない場合**
- テキストと話者IDが正しく指定されているか確認
- 必要な依存関係がすべてインストールされているか確認


