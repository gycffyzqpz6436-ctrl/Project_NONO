# NONO Dataset Workflow

## 現在の仕組み

Golden Datasetは、人間が内容を確認して承認した会話だけを置く
`dataset/jsonl/*.jsonl` です。DB用JSONLはカテゴリ、計画、レビュー情報などを
保持し、学習用JSONLはQLoRAが読む `id` と `messages` に絞ります。

50件単位に固定することで、ID範囲、カテゴリ配分、レビュー、Git履歴を追跡しやすく
しています。候補は必ずTXTで人間が読める状態を残し、明示承認まではGoldenへ入り
ません。生成・類似検査はすべてローカルで行い、外部API、APIキー、通信費に依存し
ません。会話本文は人間またはCodexがレビューTXTへ書きます。

## 普段の最短手順

### 方法A: Codexへ頼む

> 次のNONOデータセット50件を作成してください。レビューまで行い、結果を提示して
> ください。承認とpushはまだしないでください。

Codexは `AGENTS.md` に従い、prepare、TXT記入、review、修正を進めます。

### 方法B: PowerShell

```powershell
.\scripts\new_dataset_batch.ps1 -OpenInNotepad
```

または以下を直接使います。

```powershell
python -m scripts.dataset_cycle prepare --count 50
```

表示された `.instructions.md` をCodexへ渡し、対応するTXTの `User:` と `NONO:`
だけを埋めます。

## レビュー手順

レビューTXTは `dataset/candidates/review/`、候補JSONLは
`dataset/candidates/`、詳細レポートは `dataset/reports/` に作られます。

```powershell
python -m scripts.dataset_cycle review dataset/candidates/review/<draft>.txt
```

- `pass`: 自動検査で問題なし
- `warning`: 類似、文体、構成などを人間が再確認
- `reject`: ID、重複、形式など承認不能

修正箇所だけの指示書は次で作れます。

```powershell
python -m scripts.dataset_cycle repair dataset/candidates/review/<draft>.txt
```

TXTを直接直して再reviewします。ChatGPTへ貼る場合も、instructionsとTXTを渡し、
計画欄やIDを変えずUser/NONOだけ直すよう指定します。JSONLから戻す必要がある場合は
`python -m scripts.export_review_text <candidate-jsonl>` を使います。

## 承認手順

pass 50、warning 0、reject 0を確認し、人間が明示的に確定した後だけ実行します。

```powershell
.\scripts\approve_dataset_batch.ps1 `
  -ReviewFile "dataset/candidates/review/<draft>.txt" `
  -Reviewer "yuto" `
  -Push
```

確認画面で正確に `APPROVE` と入力しない限り、Golden、commit、pushは行われません。

## Git操作

approveは対象のDB用JSONL、学習用JSONL、状態ファイルだけをaddし、
`dataset: add NONO conversations <range>` でcommitします。`-Push` がある場合だけ
pushします。push失敗時もJSONLとcommitは残るため、接続や認証を直して
`git push` を再実行します。

誤承認時は履歴を破壊するresetを避け、追加ファイルを取り消す新しいcommitを作るか、
共有済みならrevertを使います。対象範囲を確認してから実施してください。

## トラブルシューティング

- `000150`重複: 内容が完全一致する既知重複は解析時だけ安全に統合されます。内容が
  異なれば停止するため、元ファイルを確認してください。
- 49件/51件: バッチは50件ちょうどに直します。部分保存・部分承認はできません。
- ID衝突: 最新Goldenからprepareをやり直します。IDを手で変更しません。
- 類似警告: User、状況、冒頭、オチを別方向へ変更してreviewを再実行します。
- `.venv`なし: `python -m venv .venv` 後、依存関係を導入します。
- RapidFuzzなし: `.venv\Scripts\python -m pip install -r requirements-generation.txt`
- push失敗: 成果物とcommitは残ります。原因解消後に `git push`。
- TXT崩れ: `#6桁ID`、`User:`、`NONO:` と区切り線を復元します。
- 状態不一致: `python -m scripts.analyze_dataset`。状態は常に実Goldenから再計算されます。

## 次回セッション用プロンプト

> Project_NONOで次のNONOデータセット50件を作成してください。AGENTS.mdと
> docs/NONO_DATASET_RUNBOOK.mdに従い、dataset_cycle prepareから開始し、
> instructions.mdの計画を守ってUser/NONOだけを記入してください。reviewを行い、
> pass 50・warning 0・reject 0まで修正し、TXTとレポートを提示してください。
> 私が明示的に承認するまではapprove、commit、pushをしないでください。

## 内部コマンド

```powershell
python -m scripts.analyze_dataset
python -m scripts.dataset_cycle prepare --count 50
python -m scripts.dataset_cycle review <review-txt>
python -m scripts.dataset_cycle repair <review-txt>
python -m scripts.dataset_cycle approve <review-txt> --reviewer yuto --commit --push
```
