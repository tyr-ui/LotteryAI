Changelog

このプロジェクトの主な変更を記録します。

形式は Keep a Changelog を参考にし、バージョン番号はセマンティック・バージョニングに準拠します。

2.0.0 - 2026-08-02

Added

• LOTO6、LOTO7、ミニロト、Numbers3、Numbers4の全5ゲーム対応
• 組合せ型ゲームと桁順序型ゲームを分けた予測・評価処理
• Numbers3の000〜999、Numbers4の0000〜9999の全候補評価
• Numbersの先頭ゼロ保持、重複桁対応、Straight／Box評価
• ゲーム別Optimizer Experienceの保存・再利用
• Base／Experience／Random／Local／Evolutionの複数探索元
• Adaptive Search Allocation
• Adaptive Evolution
• Feature Ablation
• Feature Memory／Feature Memory Analysis
• Learning Weights／Learning Strength
• 前回予測の事後評価
• 全5ゲームの予測JSON、評価JSON、レビュー用JSON出力
• common.pyによる小規模共通関数の集約
• storage.pyによるJSON読込・保存・原子的保存
• optimizer_experience_store.pyによるExperience永続化処理の分離
• optimizer_experience_stats.pyによるExperience統計処理の分離
• optimizer_adaptation.pyによる適応判定処理の分離
• GitHub ActionsのQuick CheckとFull Run
• unittestによる自動テスト一式

Changed

• LOTO6専用の試作構成から、全5ゲームを同一プロジェクトで扱う構成へ移行
• Numbersの予測方式を確率的候補生成ではなく全候補スコアリング方式へ統一
• Optimizerの固定探索数を履歴に基づく適応配分へ変更
• Experience schemaを1.3へ更新
• Experience不足時に不足枠をRandomへ補填する方式へ変更
• run_pipeline.pyとreview_output.pyのJSON入出力を共通ストレージ処理へ移行
• READMEを現行の全5ゲーム構成へ全面更新

Fixed

• Numbersのordered pairで左右位置を区別するよう修正
• Numbersの最終候補をスコア降順で出力するよう修正
• Experience保存処理がFull Runから呼ばれていなかった問題を修正
• Adaptive Search AllocationがOptimizerへ未接続だった問題を修正
• Numbers3／Numbers4のExperience保存・復元を追加
• Experience統計分割時に不足していたConfig正規化処理を補完
• Experience JSONを原子的に保存し、途中終了時の破損リスクを低減

Tested

• 全Pythonファイルの構文チェック
• 全5ゲームの定義
• データ正規化と異常検出
• 組合せ型候補の範囲・個数・重複
• Numbersの全候補列挙、先頭ゼロ、重複桁
• 出力ファイル契約
• Experience schemaと探索配分
• モックを使った全5ゲームのPipeline Smoke Test
• 全5ゲームの本番Full Run

Not adopted

• 最大値割りによるNumbers正規化v1
• パーセンタイル順位によるNumbers正規化v2

いずれもA/Bバックテストで主要指標の改善を確認できなかったため、本番共通ロジックには採用していません。

Deferred to v2.1

• schema 1.2から1.3への詳細な移行テスト
• Experience保存→再読込→候補復元の一連テスト
• Adaptive Search Allocationの分岐別テスト
• 出力パス管理方針の全体統一
• _config_value()の共通化
• normalize_float()の契約統一
• 大型ファイルの追加分割
• 不採用実験コードのexperiments/移動
• Full Runのプロファイリングと高速化
• 長期評価ダッシュボード

[1.x] - 試作期間

Added

• LOTO6を中心とした初期予測・バックテスト機能
• 頻度、遅延、ペア、形状特徴を使った候補生成
• 基本的なJSON出力とOptimizer試作