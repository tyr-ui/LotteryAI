# LotteryAI
Private
# LotteryAI

ロト6・ロト7の過去データを用いて、複数の予測アルゴリズムをバックテストし、毎回の次回候補をJSONとレポートで出力する研究プロジェクト。

## 目的

単発の勘予想ではなく、過去データに対して比較的成績が良いアプローチを継続的に検証する。

主目的は以下。

- ロト6・ロト7の公式データを取得する
- 欠損・重複・範囲外を検証する
- 頻度・遅延・ペア・トリプル・分布を特徴量化する
- 複数モデルをロールフォワードでバックテストする
- 平均一致数、2個以上一致率、3個以上一致率を比較する
- 最も成績が良いモデルで次回5パターンを出力する

## 対象

- ロト6：1〜43から6個
- ロト7：1〜37から7個

## ディレクトリ構成

```text
LotteryAI/
├── data/
│   ├── raw/
│   │   ├── loto6.csv
│   │   └── loto7.csv
│   └── processed/
│       ├── loto6_clean.csv
│       └── loto7_clean.csv
├── src/
│   ├── download.py
│   ├── validate.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   ├── optimizer.py
│   ├── predict.py
│   ├── report.py
│   └── main.py
├── output/
│   ├── prediction_loto6.json
│   ├── prediction_loto7.json
│   ├── backtest_summary.csv
│   └── report.html
├── requirements.txt
└── README.md