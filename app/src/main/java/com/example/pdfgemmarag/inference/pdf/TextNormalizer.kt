package com.example.pdfgemmarag.inference.pdf

import java.text.Normalizer

/**
 * Canonicalises extracted PDF text for search and embeddings.
 *
 * ToUnicode maps in the wild (and Skia-generated PDFs on the emulator) hand back CJK radicals in
 * place of the unified ideographs they look like: 日 -> U+2F47 (Kangxi radical), 民 -> U+2EA0 (CJK
 * Radicals Supplement). NFKC folds Kangxi radicals, fullwidth ASCII, ligatures and compatibility
 * ideographs, but CJK Radicals Supplement and CJK Strokes have no decomposition, so those come from
 * Unicode's `EquivalentUnifiedIdeograph.txt` (only the entries NFKC does not already cover).
 */
object TextNormalizer {

    fun normalize(raw: String): String {
        if (raw.none { it.code > 0x7F }) return raw
        val nfkc = Normalizer.normalize(raw, Normalizer.Form.NFKC)
        if (nfkc.none { it.code in 0x2E80..0x31EF }) return nfkc
        val sb = StringBuilder(nfkc.length)
        for (ch in nfkc) {
            val replacement = RADICAL_TO_IDEOGRAPH[ch]
            if (replacement != null) sb.appendCodePoint(replacement) else sb.append(ch)
        }
        return sb.toString()
    }

    /** src:dst hex pairs generated from EquivalentUnifiedIdeograph-17.0.0.txt. */
    // Sources are BMP radicals, but several equivalent unified ideographs are supplementary
    // code points. Keep destinations as code points rather than UTF-16 code units.
    private val RADICAL_TO_IDEOGRAPH: Map<Char, Int> = listOf(
        "2E81:5382,2E82:4E5B,2E83:4E5A,2E84:4E59,2E85:4EBB,2E86:5182,2E87:20628,2E88:5200,2E89:5202,2E8A:535C,2E8B:353E,2E8C:5C0F",
        "2E8D:5C0F,2E8E:5140,2E8F:5C23,2E90:5C22,2E91:21BC2,2E92:5DF3,2E93:5E7A,2E94:5F51,2E95:2B739,2E96:5FC4,2E97:5FC3,2E98:624C",
        "2E99:6535,2E9B:65E1,2E9C:65E5,2E9D:6708,2E9E:6B7A,2EA0:6C11,2EA1:6C35,2EA2:6C3A,2EA3:706C,2EA4:722B,2EA5:722B,2EA6:4E2C",
        "2EA7:725B,2EA8:72AD,2EA9:738B,2EAA:24D14,2EAB:76EE,2EAC:793A,2EAD:793B,2EAE:25AD7,2EAF:7CF9,2EB0:7E9F,2EB1:7F53,2EB2:7F52",
        "2EB3:34C1,2EB4:5197,2EB5:2626B,2EB6:7F8A,2EB7:2634C,2EB8:2634B,2EB9:8002,2EBA:8080,2EBB:807F,2EBC:8089,2EBD:26951,2EBE:8279",
        "2EBF:8279,2EC0:8279,2EC1:864E,2EC2:8864,2EC3:8980,2EC4:897F,2EC5:89C1,2EC6:89D2,2EC7:278B2,2EC8:8BA0,2EC9:8D1D,2ECA:27FB7",
        "2ECB:8F66,2ECC:8FB6,2ECD:8FB6,2ECE:8FB6,2ECF:9091,2ED0:9485,2ED1:9577,2ED2:9578,2ED3:957F,2ED4:95E8,2ED5:28E0F,2ED6:961D",
        "2ED7:96E8,2ED8:9752,2ED9:97E6,2EDA:9875,2EDB:98CE,2EDC:98DE,2EDD:98DF,2EDE:2967F,2EDF:98E0,2EE0:9963,2EE1:29810,2EE2:9A6C",
        "2EE3:9AA8,2EE4:9B3C,2EE5:9C7C,2EE6:9E1F,2EE7:5364,2EE8:9EA6,2EE9:9EC4,2EEA:9EFE,2EEB:6589,2EEC:9F50,2EED:6B6F,2EEE:9F7F",
        "2EEF:7ADC,2EF0:9F99,2EF1:9F9C,2EF2:4E80,31C6:200CC,31CF:4E40,31D0:4E00,31D1:4E28,31D2:4E3F,31D3:4E3F,31D4:4E36,31D5:200CD",
        "31D6:4E5B,31D7:200CA,31D8:200CE,31D9:2010C,31DA:4E85,31DB:21FE8,31DC:200CB,31DD:4E40,31DE:200D1,31DF:4E5A,31E0:4E59,31E1:2010E",
    ).joinToString(",").split(',').associate { entry ->
        val (src, dst) = entry.split(':')
        src.toInt(16).toChar() to dst.toInt(16)
    }
}
