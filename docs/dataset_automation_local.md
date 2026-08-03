# Local Dataset Automation

Project NONOのデータセット作成は、外部生成APIを使わないローカル工程です。
会話本文は人間またはCodexとの対話で作成し、自動処理は分析、テンプレート、
重複検査、形式検査、承認用変換だけを担当します。

## Setup

```powershell
python -m pip install -r requirements-generation.txt
```

必要な追加依存はYAML読込みとRapidFuzzだけです。OpenAI SDK、APIキー、
ネット接続、GPUは不要です。`sentence-transformers`は必須ではなく、
モデルの自動ダウンロードも行いません。

## Codexへの一度の依頼で承認直前まで進める

Codexへ「次の50件作って」と依頼すると、ルートの`AGENTS.md`に従って
`prepare`、TXTとinstructionsの生成、50件の執筆、review、本文修正、
再reviewを同じタスク内で行います。

再reviewでは、元のレビューTXTやGoldenを上書きせず、そのTXTから生成された
候補JSONLとレビュー結果だけを置換します。

```powershell
python -m scripts.dataset_cycle review <review-txt> --replace-results
```

`pass 50`、`warning 0`、`reject 0`になったらTXT全文とレビュー結果を提示して
停止します。`approve`、commit、pushは、人間が明示的に承認するまで
実行されません。

## 1. Golden Datasetを分析

```powershell
python -m scripts.analyze_dataset
```

以下を出力します。

- `dataset/reports/dataset_analysis.json`
- `dataset/reports/dataset_analysis.md`

ID・本文重複、カテゴリ、冒頭とオチ、文型、問い返し率、平均文字数、
直近50件の偏り、次回IDと予定範囲を確認できます。

読込みと採番だけを確認する場合:

```powershell
python -m scripts.generate_dataset --dry-run
```

この互換コマンドはローカルファイルを読むだけで、通信も出力作成もしません。

## 2. 50件のレビュー枠を作成

```powershell
python -m scripts.create_dataset_draft --count 50
```

`dataset/category_plan.yaml`の目標比率と既存件数から不足カテゴリを優先し、
次のようなファイルを作ります。

`dataset/candidates/review/nono_draft_000301_000350_<batch-id>.txt`

このTXTの`User:`と`NONO:`を人間またはCodexとの対話で記入します。
50件未満のテンプレートは作成できません。

## 3. TXTを候補JSONLへ変換

```powershell
python -m scripts.import_review_text `
  dataset/candidates/review/nono_draft_000301_000350_<batch-id>.txt
```

50件、連番ID、空欄、messages構造、ID衝突、完全一致、Unicode・空白・
句読点・全半角の正規化一致を検査します。文字n-gram、RapidFuzz、
キーワード、カテゴリ、状況、オチによる類似候補は警告とメタデータとして
残します。出力候補は`status=draft`です。

## 4. 人間向け品質レビュー

```powershell
python -m scripts.review_candidates `
  dataset/candidates/nono_candidates_000301_000350_<batch-id>.jsonl
```

各候補について類似会話、カテゴリ件数、冒頭・オチの重複、煽り表現、
問い返し、文字数、NONO基本構成を表示し、`pass`、`warning`、
`reject`に分類します。自動承認は行いません。

## 5. 必要ならTXTへ戻して修正

```powershell
python -m scripts.export_review_text `
  dataset/candidates/nono_candidates_000301_000350_<batch-id>.jsonl `
  --include-plan
```

修正したTXTは再び`import_review_text`へ渡せます。

## 6. 人間が最終承認

```powershell
python -m scripts.approve_candidates `
  dataset/candidates/nono_candidates_000301_000350_<batch-id>.jsonl `
  --database-output dataset/database/nono_database_000301_000350.jsonl `
  --training-output dataset/jsonl/nono_dataset_000301_000350.jsonl `
  --reviewer yuto
```

承認直前に最新の全Golden JSONLを再読込みし、50件、UTF-8 JSONL、
messages構造、ID衝突、完全一致、正規化一致、ローカル類似度を再検査します。
類似警告がある場合は停止します。人間が内容を確認して受容する場合に限り、
`--accept-similarity-warnings`を明示できます。

## API版からの移行

旧`generate_dataset.py`にあったOpenAIクライアント、Structured Outputs、
Embeddings、LLM Judge、APIキー確認、トークン集計は削除しました。
同名コマンドは`--dry-run`専用のローカル互換コマンドです。APIコードや
秘密情報はアーカイブしていません。
