# VV Core Inference Simple - 推論処理解析レポート

## 概要

VV Core Inference Simpleは、VOICEVOX系の音声合成エンジンのコア部分を実装したシステムです。ONNXモデルを使用して日本語テキストから音声を生成する推論パイプラインを提供します。

## システム構成

### 主要ファイル構成

- `run.py` - メインエントリーポイント
- `vv_core_inference/` - コア推論モジュール
  - `forwarder.py` - 推論統合クラス
  - `onnx_*_forwarder.py` - 各ONNXモデルのラッパー
  - `full_context_label.py` - 音韻解析処理
  - `acoustic_feature_extractor.py` - 音響特徴量処理

### 使用されるONNXモデル

1. **Yukarin S** (`model/yukarin_s/duration.onnx`)
   - 音素継続時間予測モデル
   - 入力: 音素系列、話者ID
   - 出力: 各音素の継続時間

2. **Yukarin SA** (`model/yukarin_sa/intonation.onnx`)
   - 音高(F0)予測モデル
   - 入力: モーラ情報、アクセント情報、話者ID
   - 出力: F0系列

3. **Yukarin SoSoA** (`model/yukarin_sosoa/spectrogram.onnx`)
   - スペクトログラム生成モデル
   - 入力: 音素系列、F0、話者ID
   - 出力: メルスペクトログラム

4. **HiFiGAN** (`model/hifigan/vocoder.onnx`)
   - ボコーダー（音声波形生成）
   - 入力: メルスペクトログラム
   - 出力: 音声波形

## 推論パイプライン詳細解析

### 1. テキスト前処理フェーズ

**実装場所**: `full_context_label.py:extract_full_context_label()`

```
テキスト → pyopenjtalk → フルコンテキストラベル → 音韻構造解析
```

**処理内容**:
- pyopenjtalkを使用してテキストを音韻に変換
- フルコンテキストラベルから音素、モーラ、アクセント句、発話を階層的に構築
- アクセント情報、音素境界情報を抽出

**主要クラス構造**:
```
Utterance (発話)
├── BreathGroup (息継ぎ区間)
│   └── AccentPhrase (アクセント句)
│       └── Mora (モーラ)
│           └── Phoneme (音素)
```

### 2. 音素継続時間予測フェーズ

**実装場所**: `forwarder.py:98-119`, `onnx_yukarin_s_forwarder.py`

**処理フロー**:
1. 音素文字列を音素IDに変換 (`OjtPhoneme`クラス使用)
2. Yukarin Sモデルに入力:
   - `phoneme_list`: 音素ID系列
   - `speaker_id`: 話者ID
3. 各音素の継続時間を取得
4. レート変換（200Hz基準）

**重要な処理**:
- 文頭・文末の音素継続時間を0.1に固定 (`phoneme_length[0] = phoneme_length[-1] = 0.1`)
- 継続時間を200Hzレートに正規化

### 3. 音高予測フェーズ

**実装場所**: `forwarder.py:120-161`, `onnx_yukarin_sa_forwarder.py`

**処理フロー**:
1. 音素をモーラ単位に分割 (`split_mora()`)
2. アクセント情報を母音位置にマッピング
3. Yukarin SAモデルに入力:
   - `vowel_phoneme_list`: 母音音素系列
   - `consonant_phoneme_list`: 子音音素系列  
   - `start_accent_list`: アクセント開始情報
   - `end_accent_list`: アクセント終了情報
   - `start_accent_phrase_list`: アクセント句開始情報
   - `end_accent_phrase_list`: アクセント句終了情報
   - `speaker_id`: 話者ID
4. F0値系列を取得
5. 無声音素のF0を0に設定

### 4. 特徴量統合・リサンプリングフェーズ

**実装場所**: `forwarder.py:162-171`

**処理内容**:
1. 音素系列を継続時間に基づいて展開
2. F0系列をモーラ継続時間に基づいて展開
3. 両系列を24000Hz/256フレームレートにリサンプリング
4. 音素系列をワンホットエンコーディングに変換

### 5. スペクトログラム生成フェーズ

**実装場所**: `forwarder.py:196-208`, `onnx_decode_forwarder.py`

**処理フロー**:
1. 音素を`OjtPhoneme`形式でワンホットエンコーディング
2. Yukarin SoSoAモデルに入力:
   - `f0`: F0系列
   - `phoneme`: 音素ワンホット行列
   - `speaker_id`: 話者ID
3. メルスペクトログラムを生成

### 6. 音声波形生成フェーズ

**実装場所**: `onnx_decode_forwarder.py:42-47`

**処理フロー**:
1. HiFiGANモデルに入力:
   - `spec`: メルスペクトログラム
2. 24kHzサンプリングレートの音声波形を生成

## アーキテクチャ設計の特徴

### 1. モジュラー設計

各ONNXモデルは独立したforwarder関数でラップされており、`Forwarder`クラスが統合インターフェースを提供します。これにより個別のモデル交換が容易になっています。

### 2. 音韻処理の階層化

音韻情報を`Phoneme` → `Mora` → `AccentPhrase` → `BreathGroup` → `Utterance`の階層で管理し、各レベルでの文脈情報を適切に処理しています。

### 3. ONNX Runtime活用

推論処理は全てONNX Runtimeを使用し、CPU/GPU両対応で効率的な推論を実現しています。

### 4. データフロー最適化

音素・F0・継続時間情報を段階的に構築し、最終的に統合してスペクトログラム・音声生成に渡す効率的なパイプラインです。

## 性能特性

### サンプリングレート
- 内部処理: 200Hz (音素・F0レベル)
- 音響特徴量: 24000Hz/256 ≈ 93.75Hz
- 出力音声: 24kHz

### 話者対応
- 複数話者対応（speaker_id指定）
- F0話者IDとスペクトログラム話者IDを独立指定可能

### GPU対応
- ONNX Runtime経由でCUDA対応
- デバイス選択は起動時に指定

## まとめ

本システムは、現代的な音声合成パイプラインの典型的な実装で、テキスト解析から音声生成まで4段階のニューラルネットワークモデルを組み合わせています。特に日本語の音韻特性（モーラ、アクセント）を適切に処理し、高品質な音声合成を実現する設計となっています。

コードの構造は保守性と拡張性を重視したモジュラー設計で、各段階での中間結果の取得も可能な柔軟な実装となっています。