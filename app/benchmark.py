"""第1回 課題: 応答時間とトークン数の計測・モデル比較・レート制限の考察レポート生成スクリプト。

既存の `app/ask.py` を一切変更せず、デコレータを用いてAPI呼び出しの計測と
レート制限（429 RESOURCE_EXHAUSTED）のハンドリングを行い、
結果の比較表・プロット画像（PNG）・考察を含むMarkdownレポートを自動生成します。

使い方:
    uv run python -m app.benchmark
    uv run python -m app.benchmark --output-dir reports --sleep 5
    uv run python -m app.benchmark --models gemini-3.5-flash-lite gemini-3.5-flash
"""

from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import errors

# 既存の講義モジュールから設定と定数をインポート（元ファイルは無変更）
from app.ask import DEFAULT_QUESTION
from app.config import DEFAULT_MODEL, api_error_message, get_api_key

# =====================================================================
# 1. 実験定義（モデル × 質問マトリクス）
# =====================================================================

MODELS: list[str] = [
    DEFAULT_MODEL,
    "gemini-3.5-flash",
]

QUESTIONS: list[tuple[str, str]] = [
    ("質問1 (講座固有)", DEFAULT_QUESTION),
    ("質問2 (短答・事実)", "日本の首都はどこですか？一言で答えてください。"),
    (
        "質問3 (長文・解説)",
        "LoRA（Low-Rank Adaptation）の仕組みとメリットを、初学者向けに具体例を交えて300字程度で解説してください。",
    ),
]


# =====================================================================
# 2. データ構造 & 計測・リトライ用デコレータ
# =====================================================================

@dataclass
class MetricsResult:
    """1回のAPI呼び出しの計測結果を保持するデータクラス。"""
    model: str
    question_label: str
    question_text: str
    response_text: str
    elapsed_sec: float
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    total_tokens: int
    tokens_per_sec: float
    retry_count: int = 0
    error_msg: str | None = None


def _extract_token_counts(usage: Any) -> tuple[int, int, int, int]:
    """usage_metadata から入力・出力・思考・合計トークン数を取得する。"""
    if usage is None:
        return 0, 0, 0, 0

    in_tokens = usage.prompt_token_count or 0
    out_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count or 0
    thought_tokens = getattr(usage, "thoughts_token_count", None) or 0

    # 一部のモデル/APIレスポンスでは thoughts_token_count が 0/未設定になる。
    # 合計値に含まれる思考分を差分から補完し、0 と誤表示しないようにする。
    if thought_tokens == 0 and total_tokens > 0:
        tool_tokens = getattr(usage, "tool_use_prompt_token_count", None) or 0
        thought_tokens = max(
            0, total_tokens - in_tokens - out_tokens - tool_tokens
        )

    return in_tokens, out_tokens, thought_tokens, total_tokens


def with_retry_and_rate_limit(
    max_retries: int = 3, base_wait: float = 15.0
) -> Callable:
    """無料枠のレート制限 (429 RESOURCE_EXHAUSTED) に対応するリトライデコレータ。"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, int]:
            retries = 0
            while True:
                try:
                    result = func(*args, **kwargs)
                    return result, retries
                except errors.APIError as e:
                    # 429 (Rate Limit) または 503 (一時的な過負荷) の場合は待機してリトライ
                    if e.code in (429, 503) and retries < max_retries:
                        retries += 1
                        wait_time = base_wait * retries
                        print(
                            f"  [!] API制限/一時エラー (Code {e.code}) を検知。"
                            f"{wait_time:.0f}秒待機してリトライします ({retries}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
        return wrapper
    return decorator


def measure_performance(func: Callable) -> Callable:
    """API呼び出し関数の実行時間とusage_metadataを自動抽出する計測デコレータ。"""
    @functools.wraps(func)
    def wrapper(
        client: genai.Client, model: str, question_label: str, question_text: str
    ) -> MetricsResult:
        start_time = time.perf_counter()
        try:
            (response, retries) = func(
                client, model, question_label, question_text
            )
            elapsed = time.perf_counter() - start_time

            usage = getattr(response, "usage_metadata", None)
            in_tokens, out_tokens, thought_tokens, total_tokens = (
                _extract_token_counts(usage)
            )

            # スループット: 生成されたトークン（出力＋内部思考）÷ 経過時間
            generated_tokens = out_tokens + thought_tokens
            tps = generated_tokens / elapsed if elapsed > 0 else 0.0

            return MetricsResult(
                model=model,
                question_label=question_label,
                question_text=question_text,
                response_text=response.text or "",
                elapsed_sec=elapsed,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                thought_tokens=thought_tokens,
                total_tokens=total_tokens,
                tokens_per_sec=tps,
                retry_count=retries,
            )
        except errors.APIError as e:
            elapsed = time.perf_counter() - start_time
            msg = api_error_message(e.code, e.message)
            print(f"  [×] エラー発生: {msg}")
            return MetricsResult(
                model=model,
                question_label=question_label,
                question_text=question_text,
                response_text="",
                elapsed_sec=elapsed,
                input_tokens=0,
                output_tokens=0,
                thought_tokens=0,
                total_tokens=0,
                tokens_per_sec=0.0,
                error_msg=f"Code {e.code}: {e.message}",
            )

    return wrapper


# デコレータを多重適用: 外側で時間とメタデータを計測し、内側で429リトライを制御
@measure_performance
@with_retry_and_rate_limit(max_retries=3, base_wait=20.0)
def call_gemini_api(
    client: genai.Client, model: str, question_label: str, question_text: str
) -> Any:
    """素のLLMを呼び出すコア関数（app.ask.ask と同等の呼び出し）。"""
    return client.models.generate_content(model=model, contents=question_text)


# =====================================================================
# 3. プロット画像の生成（全評価項目を1枚の画像に統合）
# =====================================================================

def _configure_japanese_font() -> None:
    """Codespace (Linux) / Mac / Windows 環境で利用可能な日本語フォントを自動設定する。"""
    import importlib.util

    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # OSに日本語フォントが入っていない環境でも、依存パッケージに同梱された
    # IPAexGothicを使えるようにする（パッケージ自体は実行しない）。
    package_spec = importlib.util.find_spec("japanize_matplotlib")
    if package_spec and package_spec.submodule_search_locations:
        bundled_font = (
            Path(next(iter(package_spec.submodule_search_locations)))
            / "fonts"
            / "ipaexg.ttf"
        )
        if bundled_font.is_file():
            fm.fontManager.addfont(str(bundled_font))
            plt.rcParams["font.family"] = "IPAexGothic"
            return

    candidates = [
        "Noto Sans CJK JP",
        "IPAGothic",
        "IPAexGothic",
        "VL Gothic",
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            return

    # システムパスから直接フォントファイルを探索 (Ubuntu/Debian系Codespace対策)
    font_dirs = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts")]
    preferred_paths: list[Path] = []
    other_paths: list[Path] = []
    for d in font_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for font_path in p.rglob("*.[to]tf"):
            name = font_path.name.lower()
            if any(k in name for k in ("noto", "cjk", "ipa", "gothic")):
                (preferred_paths if "cjk" in name or "ipa" in name else other_paths).append(
                    font_path
                )

    for font_path in [*preferred_paths, *other_paths]:
        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = prop.get_name()
        return


def generate_benchmark_plot(results: list[MetricsResult], output_path: Path) -> bool:
    """各評価項目ごとに、全モデルの質問ごとの評価結果値を並べたグループ化棒グラフを生成する。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # GUIなしのCodespace環境向けバックエンド
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[!] matplotlib がインストールされていないため、プロット生成をスキップします。")
        print("    (`uv add matplotlib` を実行するとグラフ出力が有効になります)")
        return False

    _configure_japanese_font()
    plt.rcParams["axes.unicode_minus"] = False

    # ソート順の維持（モデル順・質問順）
    models = list(dict.fromkeys(r.model for r in results))
    q_labels = list(dict.fromkeys(r.question_label for r in results))

    # 6つの評価項目を 2行×3列 のサブプロットとして1ファイルに描画
    metrics_specs: list[tuple[str, str, str]] = [
        ("elapsed_sec", "応答時間 (秒)", "{:.2f}"),
        ("input_tokens", "入力トークン数", "{:.0f}"),
        ("output_tokens", "出力トークン数 (回答)", "{:.0f}"),
        ("thought_tokens", "思考トークン数 (内部推論)", "{:.0f}"),
        ("total_tokens", "合計トークン数", "{:.0f}"),
        ("tokens_per_sec", "生成速度 (Tokens / sec)", "{:.1f}"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    fig.suptitle(
        "Gemini API モデル × 質問別 パフォーマンス比較ベンチマーク",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(q_labels))
    width = 0.8 / max(len(models), 1)
    colors = ["#2b5c8f", "#3690c0", "#d95f02", "#7570b3"]

    for idx, (attr, title, fmt) in enumerate(metrics_specs):
        ax = axes[idx // 3][idx % 3]
        max_val = 0.0

        for m_idx, model_name in enumerate(models):
            vals = []
            for q_label in q_labels:
                matched = [
                    getattr(r, attr)
                    for r in results
                    if r.model == model_name and r.question_label == q_label
                ]
                val = float(matched[0]) if matched else 0.0
                vals.append(val)
                if val > max_val:
                    max_val = val

            offset = (m_idx - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width * 0.9,
                label=model_name,
                color=colors[m_idx % len(colors)],
                edgecolor="white",
                 alpha=0.9,
            )

            # 各バーの上に数値ラベルを貼り付け
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_ylim(0, max_val * 1.22 if max_val > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


# =====================================================================
# 4. Markdownレポートの自動生成
# =====================================================================

def generate_markdown_report(
    results: list[MetricsResult],
    md_path: Path,
    plot_filename: str | None,
    models: list[str],
) -> None:
    """直積テーブル・プロット画像リンク・レート制限の考察を含むMarkdownを生成する。"""
    # モデル順 → 質問順（定義順）で厳密にソート
    model_order = {m: i for i, m in enumerate(models)}
    q_order = {q[0]: i for i, q in enumerate(QUESTIONS)}
    sorted_results = sorted(
        results,
        key=lambda r: (model_order.get(r.model, 999), q_order.get(r.question_label, 999)),
    )

    lines: list[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# Gemini API 応答時間・トークン数 比較検証レポート")
    lines.append(f"\n- **実行日時:** {now_str}")
    lines.append(f"- **検証モデル:** {', '.join(f'`{m}`' for m in models)}")
    lines.append("\n---\n")

    # 1. 質問一覧の定義
    lines.append("## 1. 検証に使用した質問セット")
    for q_label, q_text in QUESTIONS:
        lines.append(f"- **{q_label}**: 「{q_text}」")
    lines.append("\n---\n")

    # 2. 測定結果テーブル（行：(モデル, 質問) の直積2列インデックス、列：各測定指標）
    lines.append("## 2. 測定結果比較テーブル（モデル × 質問の直積）")
    lines.append("")
    lines.append(
        "| モデル名 | 質問 | 応答時間 (秒) | 入力トークン | 出力トークン | 思考トークン | 合計トークン | 生成速度 (Tok/s) | リトライ回数 |"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for r in sorted_results:
        status_note = f" (Error)" if r.error_msg else ""
        lines.append(
            f"| `{r.model}` | {r.question_label}{status_note} "
            f"| {r.elapsed_sec:.2f} "
            f"| {r.input_tokens} "
            f"| {r.output_tokens} "
            f"| {r.thought_tokens} "
            f"| {r.total_tokens} "
            f"| {r.tokens_per_sec:.1f} "
            f"| {r.retry_count} |"
        )

    lines.append("\n---\n")

    # 3. プロット画像の埋め込み
    if plot_filename:
        lines.append("## 3. 評価項目別 比較プロット")
        lines.append("")
        lines.append(f"![評価項目別プロット](./{plot_filename})")
        lines.append("\n---\n")

    # 4. 考察セクション（実測値から動的にサマリを算出して組み込む）
    lines.append("## 4. 実験結果の分析と無料枠レート制限に関する考察")
    successful_results = [r for r in sorted_results if not r.error_msg]
    error_results = [r for r in sorted_results if r.error_msg]
    total_retries = sum(r.retry_count for r in sorted_results)
    lines.append(
        f"- **実行件数:** {len(sorted_results)}（成功 {len(successful_results)}、"
        f"エラー {len(error_results)}、リトライ {total_retries}）"
    )
    lines.append("""
### (1) モデル別の応答時間と「思考トークン（Thinking Tokens）」の影響
- `thoughts_token_count` は思考**トークン数**であり、思考時間ではない。応答時間はAPIリクエスト開始からレスポンス全体の受信完了までを測定している。
- ストリーミングを使っていないため、最初のトークンが出るまでの時間（TTFT）は今回の測定対象ではない。モデル間の比較は、応答完了時間とトークン数で行う。
- 思考トークンが提供されないレスポンスでは、合計・入力・出力・ツール用トークンの差分を補助的に表示している。

### (2) 質問の性質によるトークン消費の違い（講座固有の質問 vs 一般知識）
- **質問1（講座固有の質問）**: 素のLLMは講座固有のコンテキスト（`data/glossary.json`）を持たないため、「分かりません」と短く切り上げるか、一般的な「3つのレベル」という概念を推測して長文を生成するかでモデルごとに挙動が分かれる。推測で長文を出力してしまうケースでは、ハルシネーションのリスクだけでなく、無駄な出力トークンと応答時間を消費していることが分かる。

### (3) 無料枠（Free Tier）のレート制限との関係とエンジニアリング上の示唆
Google AI Studioの無料枠には、主に以下の3つの軸でレート制限（Rate Limits）が設けられている。

| 制限指標 | 意味 | 今回の計測結果との関係・ボトルネック |
| :--- | :--- | :--- |
| **RPM** (Requests Per Minute) | 1分あたりのリクエスト回数上限 | `--sleep` でリクエスト間隔を調整する。実際の上限はモデル・プロジェクト・アカウントによって異なるため、429の発生とリトライ回数を実行結果で確認する。 |
| **TPM** (Tokens Per Minute) | 1分あたりの処理トークン数上限 | **思考トークンと長文生成が影響。** 入力プロンプトが短くても、出力トークンや `thoughts_token_count`（思考トークン）が合算されるため、長文解説や複雑な推論を連続させるとTPMを急速に消費する。 |
| **RPD** (Requests Per Day) | 1日あたりのリクエスト回数上限 | 今回の実行ではモデル数×質問数のリクエストを消費する。エージェントでは1問あたり複数回のAPI呼び出しになる場合があるため、実行回数を管理する。 |
""")

    # 5. 付録：実際のモデルの回答抜粋
    lines.append("\n---\n")
    lines.append("## 5. 各モデルの実際の回答テキスト（抜粋）")
    for r in sorted_results:
        lines.append(f"\n### `{r.model}` × {r.question_label}")
        if r.error_msg:
            lines.append(f"> **エラー:** {r.error_msg}")
        else:
            excerpt = r.response_text.strip().replace("\n", "\n> ")
            lines.append(f"> {excerpt}")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# 5. エントリポイント
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini APIの複数モデル・複数質問のベンチマークを実行し、Markdownレポートとプロットを生成する"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="レポート(benchmark_report.md)とプロット画像(benchmark_plot.png)の出力先ディレクトリ（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=4.0,
        help="無料枠のRPM制限を回避するためのリクエスト間の待機秒数（デフォルト: 4.0秒）",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="比較するモデル名（デフォルト: 現行の3モデル）",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "benchmark_report.md"
    plot_path = out_dir / "benchmark_plot.png"

    client = genai.Client(api_key=get_api_key())
    results: list[MetricsResult] = []

    models = list(dict.fromkeys(args.models))
    if not models:
        parser.error("--models には1つ以上のモデルを指定してください")

    total_runs = len(models) * len(QUESTIONS)
    current_run = 0

    print("=" * 70)
    print(f"Gemini API ベンチマークを開始します (全 {total_runs} パターン)")
    print("=" * 70)

    for model_name in models:
        for q_label, q_text in QUESTIONS:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Model: {model_name} | {q_label} を計測中...")

            res = call_gemini_api(client, model_name, q_label, q_text)
            results.append(res)

            print(
                f"  -> 完了: {res.elapsed_sec:.2f}秒 | "
                f"In: {res.input_tokens}, Out: {res.output_tokens}, "
                f"Thought: {res.thought_tokens}, Total: {res.total_tokens} tokens "
                f"({res.tokens_per_sec:.1f} tok/s)"
            )

            # 最終リクエスト以外は無料枠のレート制限回避のためにスリープを入れる
            if current_run < total_runs and args.sleep > 0:
                time.sleep(args.sleep)

    print("\n" + "=" * 70)
    print("プロット画像とMarkdownレポートを生成しています...")

    plot_created = generate_benchmark_plot(results, plot_path)
    plot_filename = plot_path.name if plot_created else None
    generate_markdown_report(results, md_path, plot_filename, models)

    print(f"✅ レポート出力完了: {md_path.resolve()}")
    if plot_created:
        print(f"✅ プロット画像出力完了: {plot_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()