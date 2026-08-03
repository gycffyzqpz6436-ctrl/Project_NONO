# Reference conversation inputs

Put human conversation reference files in this directory. The local dataset workflow
reads every nested `.jsonl`, `.json`, `.txt`, and `.md` file before review.

Reference conversations are used only to check natural User phrasing and prevent
topic reuse. They are never copied into the NONO Golden Dataset.

Supported text format:

```text
User:
...

Assistant:
...

--------------------------------------------------
```

JSON/JSONL records should use the same `messages` structure as the Golden Dataset.
