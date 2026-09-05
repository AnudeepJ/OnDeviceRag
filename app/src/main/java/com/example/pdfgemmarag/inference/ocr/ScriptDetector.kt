package com.example.pdfgemmarag.inference.ocr

/** Dominant writing system of a piece of text, used for chunk sizing and OCR recogniser routing. */
enum class Script { LATIN, CHINESE, JAPANESE, KOREAN;

    val isCjk: Boolean get() = this != LATIN
}

/**
 * Zero-dependency Unicode script analysis. Counts code points by block and picks a dominant script.
 * The OCR router calls this once per document over the first text-bearing pages; the chunker calls
 * it per paragraph.
 */
object ScriptDetector {

    data class Stats(val latin: Int, val han: Int, val kana: Int, val hangul: Int, val other: Int) {
        val letters: Int get() = latin + han + kana + hangul
        val cjk: Int get() = han + kana + hangul
        val cjkRatio: Double get() = if (letters == 0) 0.0 else cjk.toDouble() / letters
    }

    fun stats(text: CharSequence): Stats {
        var latin = 0; var han = 0; var kana = 0; var hangul = 0; var other = 0
        var i = 0
        while (i < text.length) {
            val cp = Character.codePointAt(text, i)
            i += Character.charCount(cp)
            when (Character.UnicodeScript.of(cp)) {
                Character.UnicodeScript.LATIN -> latin++
                Character.UnicodeScript.HAN -> han++
                Character.UnicodeScript.HIRAGANA, Character.UnicodeScript.KATAKANA -> kana++
                Character.UnicodeScript.HANGUL -> hangul++
                Character.UnicodeScript.COMMON, Character.UnicodeScript.INHERITED -> Unit
                else -> if (Character.isLetter(cp)) other++
            }
        }
        return Stats(latin, han, kana, hangul, other)
    }

    /**
     * @param cjkThreshold fraction of letters that must be CJK for the text to be treated as CJK.
     * Mixed technical documents (English with some Chinese) stay LATIN below this.
     */
    fun detect(text: CharSequence, cjkThreshold: Double = 0.30): Script {
        val s = stats(text)
        if (s.letters == 0 || s.cjkRatio < cjkThreshold) return Script.LATIN
        // Kana is decisive for Japanese even when Han dominates (Japanese text is mostly kanji).
        if (s.kana > 0 && s.kana >= s.hangul) return Script.JAPANESE
        if (s.hangul > s.han) return Script.KOREAN
        return Script.CHINESE
    }
}
