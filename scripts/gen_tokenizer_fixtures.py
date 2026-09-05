#!/usr/bin/env python3
"""Spike 2 helper: generate SentencePiece parity fixtures with the reference Python implementation.

Usage:
    python3 -m venv /tmp/spmenv && /tmp/spmenv/bin/pip install sentencepiece
    /tmp/spmenv/bin/python scripts/gen_tokenizer_fixtures.py ~/.cache/ondevice-rag/tokenizer.model \
        > app/src/test/resources/tokenizer_fixtures.json

The Gemma tokenizer.model is gated on HF under google/*; an ungated copy is published at
https://huggingface.co/unsloth/embeddinggemma-300m/resolve/main/tokenizer.model
(sha256 1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c).
"""
import hashlib
import json
import sys

import sentencepiece as spm

SAMPLES = [
    # Latin / European
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Straße Ärger café naïve résumé",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "El zorro marrón rápido salta sobre el perro perezoso.",
    "Revenue grew 12.5% year-over-year to $4.2M in FY2023.",
    "Table 3: Quarterly results (in thousands of USD)",
    "  leading spaces\tand\ttabs  ",
    "multiple   internal   spaces",
    "Line one\nLine two\n\nLine four",
    "e-mail: john.doe@example.com; phone: +1 (555) 010-9999",
    "UPPERCASE lowercase MiXeD CaSe",
    "C++ and C# and F# and Objective-C",
    "1234567890 0.001 1e-9 3,141,592",
    "Unicode: ☃ ★ → ∑ ∞ ≠ ≤ €",
    "Emoji test 😀🚀🍣 done",
    "title: none | text: This is a document chunk.",
    "task: search result | query: what is the capital?",
    "Curly “quotes” and ‘apostrophes’ — em dash – en dash…",
    "Ελληνικά Кириллица עברית العربية हिन्दी ไทย",
    "Ünïcödé nörmälïsätïön: ﬁ ﬂ ½ ㍿",
    "a",
    "",
    " ",
    # Chinese
    "机器学习是人工智能的一个分支。",
    "北京是中华人民共和国的首都。",
    "本公司2023年度营业收入为四百二十万元，同比增长百分之十二点五。",
    "表一：季度业绩（单位：千美元）",
    "人工智能技术正在改变我们的生活方式，从智能手机到自动驾驶汽车。",
    "繁體中文：臺灣是一個位於東亞的島嶼。",
    "中英混合 mixed text 测试 test 123 数字",
    # Japanese
    "東京都は日本の首都です。",
    "東京は日本の首都であり、最大の都市です。",
    "機械学習は人工知能の一分野である。",
    "今日はいい天気ですね。散歩に行きましょう。",
    "株式会社の売上高は前年比12.5％増の420万ドルとなった。",
    "カタカナとひらがなと漢字が混ざった文章です。",
    "全角スペース　と半角スペース の違い",
    # Korean
    "안녕하세요, 만나서 반갑습니다.",
    "기계 학습은 인공 지능의 한 분야입니다.",
    "서울은 대한민국의 수도입니다.",
    "2023년 매출은 전년 대비 12.5% 증가한 420만 달러였습니다.",
    "한국어와 English가 섞인 문장 test 123",
    # Mixed and edge cases
    "日本語とEnglishと한국어と中文が混在するテキスト。",
    "Page 250 — 第250頁 — 250ページ — 250페이지",
    "<start_of_turn>user\nhello<end_of_turn>",
    "<bos>literal bos text<eos>",
    "Question: What is the capital of Japan?\nAnswer: Tokyo.",
    "| Year | Revenue |\n| 2023 | 4.2M |\n| 2024 | 5.1M |",
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    sp = spm.SentencePieceProcessor(model_file=path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    cases = []
    for s in SAMPLES:
        ids = sp.encode(s, out_type=int)
        cases.append({"text": s, "ids": ids, "pieces": [sp.id_to_piece(i) for i in ids], "decoded": sp.decode(ids)})
    out = {
        "model_sha256": sha,
        "vocab_size": sp.get_piece_size(),
        "bos": sp.bos_id(),
        "eos": sp.eos_id(),
        "pad": sp.pad_id(),
        "unk": sp.unk_id(),
        "cases": cases,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == "__main__":
    main()
