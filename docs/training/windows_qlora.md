# WindowsでNONOをQLoRA学習・評価する

この構成は、既存の `dataset/jsonl/` を変更せず、検証、結合、3分割、
QLoRA学習、対話評価を行います。コマンドはリポジトリのルートで実行してください。

## 前提と安全なデフォルト

- Windows 11、PowerShell 7、Python 3.10または3.11を想定します。
- QLoRA学習はCUDA対応NVIDIA GPUを想定します。
- ベースモデルは、日本語を扱える比較的小型の
  `Qwen/Qwen2.5-3B-Instruct` です。利用前にモデルのライセンスも確認してください。
- RTX 5070向けに4-bit NF4、double quant、計算型 `bfloat16` を使用します。
- LoRAはrank 16、alpha 32、dropout 0.05、対象は `all-linear` です。
- 混合精度は `bf16: true`、`fp16: false` です。同時にtrueにはできません。
- 初回スモークテストは `max_steps: 5`、保存なし、validation評価なしです。
- seed 42、系列長1024、実バッチ1、勾配累積8です。
- 分割比率はtrain 0.90、validation 0.05、test 0.05です。
- 学習スクリプトはtrainとvalidationだけを読み込みます。testは学習後の最終評価用に
  保持し、学習中には一切読み込みません。
- `data/processed/`、`outputs/`、`runs/` は生成物でGit管理対象外です。
- 入力ファイルと同じパスへの出力は拒否されます。

## 1. Python環境

CUDAバージョンに合うPyTorchのインストールコマンドを
[PyTorch公式ページ](https://pytorch.org/get-started/locally/)で選んでください。

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch公式ページが表示するWindows/CUDA用コマンドを先に実行
python -m pip install -r requirements.txt
```

確認:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

`torch.cuda.is_available()` が `True` にならない場合、QLoRA学習は開始しません。
`bf16: true` の場合は `torch.cuda.is_bf16_supported()` も確認され、非対応GPUでは
モデルを読み込む前に明示的なエラーで停止します。FP16へ自動フォールバックしません。

### bitsandbytesがWindowsで動かない場合

現在のbitsandbytesはWindows上の対応CUDA/NVIDIA GPUをサポートしますが、
CUDA、PyTorch、Python、bitsandbytes wheelの組み合わせによってはDLLロードエラーや
未対応GPUエラーになることがあります。

1. 64-bit Pythonを使用していることを確認する
2. PyTorchとCUDAドライバーの互換性を確認する
3. 仮想環境を作り直し、PyTorchを先に入れてからrequirementsを入れる
4. 解決しない場合はWSL2のUbuntu/CUDA環境を使用する

`load_in_4bit: false` にすればbitsandbytes量子化を無効化できますが、それはQLoRAでは
なく通常のLoRAとなり、必要VRAMが大きく増えます。CPUでのQLoRA学習は既定の対象外です。

## 2. 元データの検証

PowerShellはPythonへワイルドカードを展開しないため、パターンを引用符で囲みます。
スクリプト自身が `*.jsonl` を展開します。

```powershell
python -m scripts.validate_jsonl "dataset\jsonl\*.jsonl" `
  --expected-start 1 --expected-end 200
```

検証内容:

- 各行のUTF-8読み込みとJSONパース
- 6桁ID、ID重複、期待範囲の欠番
- messages配列、role、空欄、先頭末尾の空白
- `system`（任意）→ `user` → `assistant` の交互順序
- assistantのみの挨拶レコード
- 同一user本文、同一assistant本文

同一本文は別の文脈で意図的に使われる可能性があるため警告扱いです。構造不正、欠番、
重複IDはエラーです。現在の元データでは、ファイル境界のID `000150` が重複しています。

## 3. ID順の結合

結合の既定動作は、同一IDが1件でもあればエラーです。現データのように同一IDの
user/assistantのrole、順序、本文が完全一致する場合だけ統合するには、
`--deduplicate-identical` を明示します。本文中の空白や改行も厳密比較します。
同一IDでもuserまたはassistantが異なる場合は常にエラーです。

```powershell
python -m scripts.merge_jsonl "dataset\jsonl\*.jsonl" `
  --output "data\processed\nono_000001_000200.jsonl" `
  --database-output "data\processed\nono_000001_000200_db.jsonl" `
  --deduplicate-identical
```

- `--output`: 学習用。`id` と `messages` だけを書き出します。
- `--database-output`: 任意。元レコードの未知フィールドを保持します。
- 軽量レコードに `status` がなければ `"unreviewed"` を追加します。
- 軽量レコードに `language` がなければ `"ja"` を追加します。
- 重複レコード間で未知フィールドの値が競合する場合、先の値を維持し、すべての値を
  `_deduplication_conflicts` に記録します。

統合後、統合したID一覧、ID件数、出力レコード件数を表示します。元ファイルは
読み取り専用で扱い、変更も削除もしません。

```powershell
python -m scripts.validate_jsonl "data\processed\nono_000001_000200.jsonl" `
  --expected-start 1 --expected-end 200
```

## 4. train/validation/test分割

既定値では200件を180 / 10 / 10へ分割します。

```powershell
python -m scripts.split_jsonl "data\processed\nono_000001_000200.jsonl" `
  --train-output "data\processed\train.jsonl" `
  --validation-output "data\processed\validation.jsonl" `
  --test-output "data\processed\test.jsonl" `
  --train-ratio 0.90 `
  --validation-ratio 0.05 `
  --test-ratio 0.05 `
  --seed 42
```

同じ入力、比率、seedなら同じ分割になります。選択はseed付き疑似乱数で行い、
各出力内はID順に戻します。端数は最大剰余法で配分し、合計件数を維持します。

## 5. QLoRA学習とvalidation評価

モデル名、量子化、LoRA、epoch、バッチ、系列長、生成設定は
`configs/qlora.yaml` で変更できます。初回はモデルのダウンロードが発生します。

```powershell
python -m scripts.train_qlora --config "configs\qlora.yaml"
```

初回設定は5 stepだけのスモークテストです。

```yaml
quantization:
  compute_dtype: bfloat16

training:
  bf16: true
  fp16: false
  max_steps: 5
  save_strategy: "no"
  eval_strategy: "no"
```

起動時にGPU名、model dtype、bitsandbytes compute dtype、BF16/FP16設定、
max_stepsを表示します。

学習データ処理:

1. JSONLの `messages` を読み込む
2. ベースモデルtokenizerの `apply_chat_template` を適用する
3. 最後のassistant応答より前を `prompt`、最後のassistant応答を `completion` にする
4. TRLの `completion_only_loss=True` によりcompletionだけをloss対象にする

この方式はchat templateのassistant-mask対応に依存せず、最終assistant応答だけを
学習対象にします。スモークテストでは `eval_strategy: "no"` のためvalidation評価を
実行しません。testファイルは設定に記載されていますが、`train_qlora.py` は
読み込みません。

スモークテストでは `save_strategy: "no"` のため途中checkpointは保存しませんが、
正常終了後のアダプターとtokenizerは `outputs/nono-qlora/` に保存されます。
CUDA out-of-memoryの場合は、まず `max_length` を512へ下げてください。

本学習へ移行する際は、例として `max_steps: -1`、`save_strategy: "epoch"`、
`eval_strategy: "epoch"` に戻すと、`epochs`の値に従って学習・validation評価できます。

## 6. CLIで会話評価

```powershell
python -m scripts.chat_cli `
  --adapter "outputs\nono-qlora" `
  --config "configs\qlora.yaml" `
  --system-prompt "prompts\system.txt"
```

- `/reset`: 会話履歴を消去
- `/exit`: 終了

生成パラメータは`configs/qlora.yaml`の`inference`で変更できます。

```yaml
inference:
  max_new_tokens: 256
  do_sample: true
  temperature: 0.8
  top_p: 0.9
  repetition_penalty: 1.05
  no_repeat_ngram_size: 3
```

`no_repeat_ngram_size: 3`により、同一3-gramの反復生成を抑制します。

GPUが使えない場合、対話CLIは4-bitを無効化してCPUへフォールバックします。
3BモデルのCPU推論は遅く、多くのRAMを使う可能性があります。

## 7. 軽量テスト

データ処理テストはモデルやGPUを必要としません。

```powershell
python -m unittest discover -s tests -v
python -m compileall -q nono_lora scripts tests
```

## JSONL仕様

各行はUTF-8のJSONオブジェクトです。

```json
{"id":"000001","messages":[{"role":"user","content":"こんにちは"},{"role":"assistant","content":"へぇ〜？♡♪（笑）\n次の行"}]}
```

- `id`: 6桁の数字文字列
- `messages`: 1件以上
- `role`: `system`、`user`、`assistant` のいずれか
- `content`: 空白だけではなく、先頭末尾に不要な空白がない文字列
- 通常会話: 先頭systemは任意、その後user/assistantが交互、末尾assistant
- assistantのみの挨拶レコードも許可
- `♡`、`♪`、`〜`、`（笑）`、本文内改行はUTF-8 JSONLで保持
- 未知の追加フィールドはDB用出力で保持

## API互換性

`requirements.txt` では、実装確認対象を次の範囲へ制限しています。

- Transformers `>=4.57,<5`
- TRL `>=0.26,<0.29`
- PEFT `>=0.17,<0.19`

Transformersの `BitsAndBytesConfig`、PEFTの `LoraConfig`、TRLの
`SFTTrainer` / `SFTConfig` を使用します。依存関係を範囲外へ更新する場合は、
学習前にAPI差分を確認し、ユニットテストと小規模dry runを再実行してください。
