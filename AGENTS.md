# Project NONO dataset workflow

「次の50件を作って」「NONOデータセットを追加して」「次のバッチを作成して」と依頼されたら、次の手順を使う。

1. `python -m scripts.dataset_cycle prepare --count 50`
2. 生成された `.instructions.md` を読み、対応するレビューTXTの `User:` と `NONO:` だけを50件埋める
3. `python -m scripts.dataset_cycle review <review-txt>`
4. warning/rejectを修正し、必要なら `repair` を使い、pass 50・warning 0・reject 0まで再検査する
5. コピペ可能なTXTとレビュー結果を人間へ提示する
6. 「確定」「承認」「OK」などの明示承認を待つ
7. 承認後だけ `dataset_cycle approve` を実行し、依頼がある場合だけ `--commit --push` を付ける

禁止事項:

- 承認前にGoldenへ追加、commit、pushしない
- 既存Goldenを書き換えない
- IDを手動採番せず、全Goldenの最大ID+1を使う
- 50件未満のバッチを作らない
- 同じ話題の単なる言い換えを採用しない
- `dataset/jsonl`、`dataset/database`、`references`、分析レポートをすべて参照する
- 場所・失敗原因・感情・回答方針・煽り・オチのいずれかが既存と同型なら作り直す
- 煽りを主成分にし、優しいAI定型句だけで終わらせない
- 問い返し率を30〜50%にし、冒頭・オチ・段落構成を連続させない
- 外部生成API、OpenAI API、APIキーを使わない
- 人間が読めるTXTとレビュー結果を省略しない
