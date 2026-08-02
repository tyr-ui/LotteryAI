LotteryAI v2

LotteryAI v2は、過去の抽せんデータを取得・検証し、複数の統計特徴量と探索方式をバックテストして、次回候補をJSONで出力する研究用プロジェクトです。

対応ゲームは以下の5種類です。

• LOTO6
• LOTO7
• ミニロト
• Numbers3
• Numbers4

宝くじの当選を保証するものではありません。目的は、再現可能な検証基盤を作り、各方式をランダム基準や過去成績と比較できる状態を維持することです。

現在の主な機能

共通

• 公式データまたは代替データの取得
• CSV正規化
• 欠損・重複・範囲外・抽せん回の検証
• ロールフォワード・バックテスト
• 複数Configの比較
• 前回予測の事後評価
• JSON形式の予測・評価・レビュー出力
• ゲーム別Optimizer Experienceの保存・再利用
• Adaptive Search Allocation
• Adaptive Evolution
• GitHub ActionsによるQuick CheckとFull Run

組合せ型ゲーム

対象：

• LOTO6
• LOTO7
• ミニロト

主な特徴量：

• 全期間頻度
• 直近頻度
• 遅延
• ペア・トリプル
• 前回数字との重複
• 奇偶・高低・連番・ブロック分布などの形状特徴

探索元：

• Base
• Experience
• Random
• Local
• Evolution

桁順序型ゲーム

対象：

• Numbers3
• Numbers4

特徴：

• 数字の順序を区別
• 同じ数字の重複を許可
• 先頭ゼロを保持
• Numbers3は000〜999を全列挙
• Numbers4は0000〜9999を全列挙

主な特徴量：

• 桁位置別頻度
• 直近の桁位置別頻度
• 桁位置別遅延
• 全体数字頻度
• ordered pair / ordered triplet
• 重複形状
• 合計・奇偶・高低
• 隣接桁差
• Straight履歴
• Box集合履歴

評価：

• Straight
• Box
• 位置一致数
• 順不同一致数
• Top-N評価
• ランダム基準との差

ゲーム設定

|ゲーム     |方式 |範囲  |選択数・桁数|重複|順序   |
|--------|---|---:|-----:|--|-----|
|LOTO6   |組合せ|1〜43|6個    |不可|区別しない|
|LOTO7   |組合せ|1〜37|7個    |不可|区別しない|
|ミニロト    |組合せ|1〜31|5個    |不可|区別しない|
|Numbers3|桁順序|0〜9 |3桁    |可 |区別する |
|Numbers4|桁順序|0〜9 |4桁    |可 |区別する |

ディレクトリ構成

```text
LotteryAI-main/
├── .github/
│   └── workflows/
│       └── run.yml
├── output/
├── src/
│   ├── main.py
│   ├── compare.py
│   ├── run_pipeline.py
│   ├── games.py
│   ├── data_loader.py
│   ├── common.py
│   ├── storage.py
│   │
│   ├── features.py
│   ├── predictor.py
│   ├── backtester.py
│   │
│   ├── numbers_features.py
│   ├── numbers_predictor.py
│   ├── numbers_backtester.py
│   ├── numbers_optimizer.py
│   │
│   ├── optimizer.py
│   ├── optimizer_search.py
│   ├── optimizer_evolution.py
│   ├── optimizer_evaluation.py
│   ├── optimizer_ablation.py
│   ├── optimizer_learning.py
│   ├── optimizer_experience.py
│   ├── optimizer_experience_store.py
│   ├── optimizer_experience_stats.py
│   ├── optimizer_adaptation.py
│   │
│   ├── feature_memory.py
│   ├── feature_memory_analyzer.py
│   └── review_output.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_games.py
│   ├── test_optimizer_experience.py
│   ├── test_output_contract.py
│   ├── test_pipeline_smoke.py
│   └── test_prediction_constraints.py
├── pytest.ini
├── requirements.txt
└── README.md
```

主要モジュール

run_pipeline.py

全5ゲームの実行を制御する正式なパイプラインです。

主な処理：

1. データ取得・正規化・検証
2. 前回予測の事後評価
3. ゲーム別Optimizer実行
4. 次回候補生成
5. Experience保存
6. 各種JSON出力

games.py

全ゲームの設定を管理します。

数字範囲、選択数、桁数、学習期間、バックテスト回数、出力ファイル名などを定義しています。

optimizer_experience.py

Experienceの公開APIと、ゲーム別履歴の正規化・保存内容生成を担当します。

低レベル処理は以下へ分離されています。

• optimizer_experience_store.py：JSON読込・原子的保存
• optimizer_experience_stats.py：Config・探索元統計
• optimizer_adaptation.py：Adaptive Evolution・探索枠配分

common.py

小規模な共通関数を管理します。

storage.py

JSON読込、通常保存、原子的保存を管理します。

セットアップ

Python 3.11を想定しています。

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

実行方法

プロジェクトルートで実行します。

```bash
python src/run_pipeline.py
```

互換入口として以下も残しています。

```bash
python src/main.py
python src/compare.py
```

正式な実行入口はsrc/run_pipeline.pyです。

Full Runは全5ゲームのデータ取得、バックテスト、探索、出力更新を行うため、実行環境によっては約30分かかります。

テスト

GitHub ActionsのQuick Checkと同じテスト：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
```

pytest.iniもあるため、pytestがインストールされている環境では以下でも実行できます。

```bash
pytest -q
```

現在のテスト対象：

• 全5ゲームの定義
• CSV正規化とデータ検証
• 組合せ型候補の範囲・個数・重複
• Numbersの全候補列挙・重複桁・先頭ゼロ
• 予測JSONの出力契約
• Experience schemaとAdaptive Search Allocation
• モックを使った全5ゲームのPipeline Smoke Test

GitHub Actions

.github/workflows/run.ymlには以下の2ジョブがあります。

Quick Check

mainブランチへのpush時に実行されます。

• 依存関係のインストール
• Python構文チェック
• 主要モジュールのimport確認
• unittest実行

Full Run

手動実行またはスケジュールで実行されます。

• 全5ゲームのパイプライン実行
• output/の更新を自動コミット
• レビュー用JSONをArtifactとして保存
• 全出力をArtifactとして保存

主な出力

```text
output/
├── prediction_optimizer_loto6.json
├── prediction_optimizer_loto7.json
├── prediction_optimizer_miniloto.json
├── prediction_optimizer_numbers3.json
├── prediction_optimizer_numbers4.json
├── prediction_box_numbers3.json
├── prediction_box_numbers4.json
├── optimizer_result.json
├── optimizer_experience.json
├── evaluation_history.json
├── evaluation_summary.json
├── feature_memory.json
├── feature_memory_analysis.json
├── learning_strength.json
├── run_summary.json
└── review_bundle.json
```

予測ファイル

各ゲームの次回候補を保存します。

Numbers3・Numbers4は先頭ゼロを保持するため、文字列表現も出力します。

optimizer_result.json

全ゲームのOptimizer結果、探索メタデータ、候補順位などを保存します。

optimizer_experience.json

ゲーム別に以下を保存します。

• history
• config statistics
• search source statistics
• adaptation
• latest

現在のschema versionは1.3です。

run_summary.json

Full Run全体の概要を保存します。

review_bundle.json

監査・レビューに必要な情報をまとめた出力です。

ExperienceとAdaptive Search Allocation

各ゲームの優秀なConfigと評価結果を履歴として保存し、次回の探索候補へ再利用します。

探索元は以下です。

• Base
• Experience
• Random
• Local
• Evolution

探索枠は固定値だけでなく、過去の探索元別成績を基に決定論的に再配分されます。

原則：

• Baseは比較基準として残す
• Experienceへ過度に集中しない
• Random枠を残す
• サンプル不足時は既定値を使う
• Experience候補不足時はRandomへ戻す
• 1回あたりの配分変更を小さくする

実験用コード

以下は正規化方式のA/B検証用で、本番予測経路から分離されています。

```text
src/numbers_predictor_rank_v2.py
src/compare_numbers_normalization.py
```

最大値割り正規化v1および順位パーセンタイル正規化v2は、本番共通ロジックとして採用していません。

将来整理する場合は、experiments/等へ移動し、不採用実験であることを明記します。

開発方針

• 新機能追加前に既存テストを通す
• 1回の変更範囲を限定する
• Quick Check成功前に次の変更へ進まない
• Phase完了時にFull Runする
• リファクタリングでは予測結果や評価値を意図せず変更しない
• 削除は呼出元と出力互換性を確認してから行う
• 根拠のない特徴量追加や探索アルゴリズム追加を避ける

完成条件

LotteryAI v2は、以下を満たすことを完成条件とします。

• 全5ゲームのデータ取得・検証
• 全5ゲームのバックテスト
• 全5ゲームの次回候補出力
• Straight / Box評価
• ゲーム別Experience保存・再利用
• Adaptive Evolution
• Adaptive Search Allocation
• 前回予測の事後評価
• JSON schemaの固定
• Quick Check成功
• 全5ゲームFull Run成功
• READMEとコードの一致
• 重大な重複・不要コード・循環importが残っていない

「予測精度が必ずランダムを上回ること」は完成条件ではありません。