# Project NONO dataset workflow

「次の50件を作って」「NONOデータセットを追加して」「次のバッチを作成して」と依頼されたら、追加指示を待たず、承認直前まで次の手順を一度のタスク内で完走する。

1. `python -m scripts.dataset_cycle prepare --count 50`
2. 生成されたレビューTXTと`.instructions.md`を読む
3. 全Golden、管理用DB、reference、`dataset/reports/dataset_analysis.json`と`.md`、直近100件を読む
4. 最近多い冒頭・煽り・問い返し・カテゴリを集計し、「今回避ける話題」を内部で決める
5. 対応するレビューTXTの`User:`と`NONO:`だけを50件埋める
6. `python -m scripts.dataset_cycle review <review-txt>`
7. warning/rejectごとに類似ID・理由・修正方向を確認し、レビューTXTの本文を修正する
8. 修正後は`python -m scripts.dataset_cycle review <review-txt> --replace-results`で、そのTXTから派生した候補・レビュー結果だけを置換して再reviewする
9. pass 50・warning 0・reject 0になるまで、修正と再reviewを繰り返す
10. コピペ可能なTXT全文と最終レビュー結果を人間へ提示する
11. この時点で停止し、「確定」「承認」「OK」などの明示承認を待つ
12. 承認後だけ`dataset_cycle approve`を実行し、依頼がある場合だけ`--commit --push`を付ける

## 生成時の必須確認

- 既存Golden全件（現在550件。増加後はその全件）を必ず読む
- `dataset/database`の管理用DBを必ず読む
- `references`を必ず読む。レコードが0件でも、0件であることを確認する
- `dataset_analysis`を必ず読む
- 直近100件を必ず確認する
- 最近多い冒頭、煽り、問い返し、カテゴリを避ける方向で分散する
- 執筆前に「今回避ける話題」を内部で決める
- 同じカテゴリでも、場所・状況・感情・回答・オチが別物の会話だけを採用する
- NONOは最初に内心を見透かし、煽りを先に置き、主導権を握る
- 優しいAI・相談員・先生口調にせず、メスガキ成分を維持する

## 絶対禁止

- 承認前に`approve`、Golden追加、commit、pushを行わない
- 既存Goldenを書き換えない
- IDを手動採番せず、全Goldenの最大ID+1を使う
- 50件未満のバッチを作らない
- 同じ話題の単なる言い換えを採用しない
- 同じ会話構造、回答方針、煽りの流れ、最後の質問を採用しない
- 場所・失敗原因・感情・回答方針・煽り・オチのいずれかが既存と同型なら作り直す
- 優しいAI定型句だけで終わらせない
- 問い返し率を30〜50%から外さない
- 冒頭・オチ・段落構成を連続させない
- 外部生成API、OpenAI API、APIキーを使わない
- 人間が読めるTXT全文とレビュー結果を省略しない
