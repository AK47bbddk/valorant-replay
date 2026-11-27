# タスク指示書：スモーク時系列持続判定・生存人数＆コントローラー構成制約の統合

- **タスクID / ブランチ**: `task/temporal-smoke-alive-constraints`
- **作成日**: 2026-11-27
- **参照すべき `PROMPT.md` の節**: `## ミニマップの配置とキルの正確さ`

## 背景・目的
1. ドームスモークは実戦で 13.5〜19.25 秒間ほぼ動かず持続するが、従来の `tactics.py` の節目スモーク検知は単一フレームのみを見ており、1 フレームの一過性ノイズ（フラッシュ・爆発・UI）や 1 フレーム欠損に弱かった。
2. `tactics.py` にはすでに `alive_fn(t)`（各時刻のチーム生存人数）と `agent_teams`（出場エージェント構成）があるにもかかわらず、`temporal.correct_agent_dots_sequence` や節目スモーク検知の制約として活かされていなかった。

## 触ってよいファイル
- `docs/tasks/temporal-smoke-alive-constraints.md`
- `vreplay/temporal.py`
- `vreplay/tactics.py`
- `tests/test_temporal.py`

## 変更内容
1. `vreplay/temporal.py`:
   - `correct_agent_dots_sequence` の `alive_counts` に `"total"`（合計生存人数）および `"warm"` / `"cool"` の上限トリミング（実測優先・高スコア優先）を追加。
   - `DOME_SMOKE_CAPS` / `max_dome_smokes_for_comp` を追加し、`correct_smokes_sequence` に静止持続時間フィルタ（`min_duration_sec`）とコントローラー構成上限（`controller_agents` / `max_active_smokes`）を追加。
2. `vreplay/tactics.py`:
   - `alive_fn` がある場合に `alive_counts`（`{"total": a1 + a2}`）を `correct_agent_dots_sequence` へ連携。
   - 節目スモーク検知で `keep[(n, lbl)]` の複数フレームを `correct_smokes_sequence(..., controller_agents=all_agents)` で時系列検証するように更新。
3. `tests/test_temporal.py`:
   - 生存人数上限・持続時間フィルタ・コントローラー構成上限のオフライン単体テストを追加。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_temporal.py tests/test_tactics.py tests/test_smokes.py
```

## 受け入れ条件
- [x] 指定テストがネットワークなしで全件通る
- [x] 既存の `correct_agent_dots_sequence` / `correct_smokes_sequence` の後方互換性を維持する
