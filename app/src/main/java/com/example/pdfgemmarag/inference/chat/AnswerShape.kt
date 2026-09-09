package com.example.pdfgemmarag.inference.chat

/**
 * How the answer should be shaped, independent of where evidence comes from ([QuestionIntent]).
 * Detected from question wording only; never from document content.
 */
enum class AnswerShape {
    /** Short fact, value or requirement. */
    DEFAULT,
    /** "What is a X", "define", "what makes a X a Y": answer with the defining criteria, never a section pointer. */
    DEFINITION,
    /** Steps, procedures, complete lists: preserve order and cardinality. */
    PROCEDURE,
    /** "Which section covers", "where in the document": a structural pointer is the answer. */
    NAVIGATION,
    /** Table row/column lookups. */
    TABLE,
    ;

    companion object {
        fun of(question: String): AnswerShape {
            val text = question.trim()
            return when {
                NAVIGATION_HINT.containsMatchIn(text) -> NAVIGATION
                DEFINITION_HINT.containsMatchIn(text) -> DEFINITION
                AnswerPolicy.isProcedural(text) -> PROCEDURE
                TABLE_HINT.containsMatchIn(text) -> TABLE
                else -> DEFAULT
            }
        }

        private val NAVIGATION_HINT = Regex(
            "(?i)\\b(?:which|what)\\s+(?:section|chapter|part|clause|appendix)\\b(?!\\s+(?:is|are)\\s+(?:a|an)\\b)|" +
                "\\bwhere\\s+(?:in\\s+the\\s+(?:document|manual|specification|pdf)|(?:is|are)\\s+.{0,60}\\b(?:covered|discussed|described|specified|located))\\b",
        )
        private val DEFINITION_HINT = Regex(
            "(?i)\\b(?:define\\b|definition\\s+of\\b|meaning\\s+of\\b|what\\s+does\\s+.{1,60}\\s+mean\\b|" +
                "what\\s+makes\\s+(?:a|an|the)\\b|" +
                "what\\s+(?:is|are)\\s+(?:a|an)\\s+(?:[\\p{L}'’-]+\\s?){1,4}\\??\\s*$|" +
                "what\\s+is\\s+(?:meant|understood)\\s+by\\b)",
        )
        private val TABLE_HINT = Regex("(?i)\\b(?:table|row|column|matrix)\\b")
    }
}
