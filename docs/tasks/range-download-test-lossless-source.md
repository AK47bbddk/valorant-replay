# range-download-test-lossless-source（range ダウンロードの輝度照合テストの生成元をロスレスに）

※この指示書は Muse（窓口）が 2026-11-11 の CI 失敗の調査結果をもとに作成。青木の承認後に実装する。範囲はテスト1件の生成元エンコードのみ。製品コードの変更は含まない（調査で製品側の不具合が見つかった場合は止めて報告する）。

## 目的
GitHub の test ワークフローが push のたびに失敗する。落ちているのは `tests/test_video_cache.py::test_range_download_matches_source` の1件だけで、マージ前のコミット（011b48e）でも同じ形で落ちる既存の失敗。2026-11-11 の CI 同等環境（`.venv-ci`、依存一式＋ffmpeg）での再現は 280 passed / 1 failed。

## 確認済みの事実（2026-11-11、VM `.venv-ci` で検証）
- 再現: `python -m pytest tests/test_video_cache.py::test_range_download_matches_source -x -q` → `1 failed in 6.43s`。失敗箇所は同テストの輝度照合（`assert all(abs(ref[t] - got[t]) < 3 for t in common)`、test_video_cache.py:140）。
- 診断（テストと同じ手順を再現し、輝度差を数えた）:
  - 時刻の対応は 72/72 件で一致（フレームの時刻ずれではない）。
  - 差が 3 以上なのは 26/72 件、最大差は 4.0。`got` は `ref` より系統的に 3〜4 低い（差が 2.0 のフレームも多数ある）。
  - マニフェストは1パートで、offset -1.933、length 40.0（要求 32→70 秒に対し、キーフレーム／フラグメント境界まで遡って取っている）。
- 原因の切り分け: 生成元のエンコードに `-qp 0`（ロスレス）を付けただけで、同じ手順の輝度差は全72件で 0.0 になった。つまり range で取った断片自体は元と同じフレームを同じ時刻に出しており、製品側の取り違えではない。差はロスありエンコードの生成元を、フラグメント化した mp4 として切り出して復号したときの量子化ノイズ（しきい値 `< 3` に対して 3〜4 のずれ）である。
- 比較: 同ファイルの `test_split_video_reads_like_one` は生成元に `-qp 0` を付けており、同じ `< 3` のしきい値で通っている。`test_range_download_matches_source` の生成元だけ `-qp` が無い。

## 規則
1. しきい値 `< 3` は変えない。テストを緩めて通さない（AGENTS.md の約束どおり）。
2. 製品コード（`vreplay/video.py`、`vreplay/ytrange.py`）は変えない。`-qp 0` で差が 0.0 になることをもって製品側に不具合なしと判断する。もし `-qp 0` でも差が出る場合は、しきい値を緩めずに止めて、その差の内容を `docs/tasks/range-download-test-lossless-source.md` の末尾の「質問」に書いて報告する。
3. 変更は `test_range_download_matches_source` の生成元エンコードのみ。通っている `test_range_download_hls_segments` など他のテストには触らない。

## 触ってよいファイル
- `tests/test_video_cache.py`（`test_range_download_matches_source` の ffmpeg 生成コマンドに `-qp 0` を追加する箇所のみ）

## やること
1. ブランチ `task/range-download-test-lossless-source` を切る（AGENTS.md の手順どおり）。
2. `test_range_download_matches_source` の生成元 ffmpeg コマンドに `-qp 0` を追加する（`test_split_video_reads_like_one` と同じ位置・同じ書き方にする。`-pix_fmt yuv420p` の並びに合わせる）。
3. 対象テストを実行し、通ることを確認する。ついでに `tests/test_video_cache.py` 全体を実行する。
4. 全体テストを実行し、失敗・エラーが増えていないことを確認する（この1件が直って 281 passed / 0 failed になる想定）。

## 通すテスト
- `python -m pytest -x -q tests/test_video_cache.py::test_range_download_matches_source`
- `python -m pytest -q tests/test_video_cache.py`
- 全体: `python -m pytest -q`（既存の失敗・エラーが増えていないこと）

## 終わりの条件
- `test_range_download_matches_source` が通り、輝度差が 0 付近であることが報告にある
- しきい値 `< 3` と製品コードに変更がない
- 全体テストで新たな失敗が出ていない

## 質問
（実装担当が書く。無ければ空のまま）
