package com.example.pdfgemmarag.inference.chat

/** Shared, dependency-free answer-shape policy used by prompting and generation limits. */
internal object AnswerPolicy {
    fun isProcedural(question: String): Boolean = PROCEDURAL_HINT.containsMatchIn(question)

    fun maxOutputTokens(intent: QuestionIntent, question: String): Int = when (intent) {
        QuestionIntent.SECTION_SUMMARY -> 288
        QuestionIntent.DOCUMENT_OVERVIEW -> 224
        else -> when (AnswerShape.of(question)) {
            AnswerShape.PROCEDURE -> 288
            AnswerShape.DEFINITION -> 200
            else -> 160
        }
    }

    private val PROCEDURAL_HINT = Regex(
        "(?i)\\b(?:steps?|procedures?|instructions?|precautions?|techniques?|rules?|guidelines?|" +
            "measures?|responsibilities|list|enumerate|priority\\s+order)\\b|" +
            "\\bhow\\s+(?:to|do|does|should|must|can)\\b|" +
            "\\bwhat\\s+(?:must|should)\\s+be\\s+done\\b|" +
            // Questions such as "What emergencies must the plan consider?" request the
            // members of a source list even though they do not contain the word "list".
            "\\bwhat\\b.{0,80}\\b(?:include|includes|consider|cover|address|contain)\\b",
    )
}
