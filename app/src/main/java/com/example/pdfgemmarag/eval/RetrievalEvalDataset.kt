package com.example.pdfgemmarag.eval

/**
 * Fixed multilingual retrieval set: 15 facts × 2 question phrasings × 7 languages = 210 questions
 * (30 per language). Pages are the gold labels for page-hit@K.
 *
 * Corpus text is a fictional Acme 2024 annual report so the harness never depends on a real PDF.
 */
object RetrievalEvalDataset {

    const val DOC_HASH = "eval-acme-2024"
    const val DOC_NAME = "Acme 2024 Annual Report (eval)"

    data class Fact(val id: String, val page: Int, val texts: Map<String, String>)
    data class Question(val id: String, val lang: String, val question: String, val expectedPages: Set<Int>, val factId: String)
    data class Chunk(val factId: String, val page: Int, val lang: String, val text: String, val chunkIndex: Int)

    val LANGS = listOf("en", "zh", "fr", "de", "it", "ja", "ko")

    val facts: List<Fact> = listOf(
        fact("revenue", 5, "en" to "Acme reported full-year revenue of 4.2 billion dollars in 2024.",
            "zh" to "Acme 2024年全年营收为42亿美元。",
            "fr" to "Acme a déclaré un chiffre d'affaires annuel de 4,2 milliards de dollars en 2024.",
            "de" to "Acme meldete 2024 einen Jahresumsatz von 4,2 Milliarden Dollar.",
            "it" to "Acme ha riportato ricavi annui di 4,2 miliardi di dollari nel 2024.",
            "ja" to "Acmeの2024年通期売上高は42億ドルでした。",
            "ko" to "Acme는 2024년 연간 매출 42억 달러를 기록했습니다."),
        fact("ceo", 12, "en" to "Chief Executive Officer Mina Park has led Acme since 2019.",
            "zh" to "首席执行官 Mina Park 自2019年起领导 Acme。",
            "fr" to "La directrice générale Mina Park dirige Acme depuis 2019.",
            "de" to "Vorstandsvorsitzende Mina Park leitet Acme seit 2019.",
            "it" to "L'amministratrice delegata Mina Park guida Acme dal 2019.",
            "ja" to "最高経営責任者のMina Parkは2019年からAcmeを率いています。",
            "ko" to "최고경영자 Mina Park는 2019년부터 Acme를 이끌고 있습니다."),
        fact("hq", 18, "en" to "Acme headquarters is in Seoul, Republic of Korea.",
            "zh" to "Acme 总部位于韩国首尔。",
            "fr" to "Le siège d'Acme se trouve à Séoul, en République de Corée.",
            "de" to "Der Hauptsitz von Acme befindet sich in Seoul, Republik Korea.",
            "it" to "La sede di Acme si trova a Seul, Repubblica di Corea.",
            "ja" to "Acmeの本社は大韓民国ソウルにあります。",
            "ko" to "Acme 본사는 대한민국 서울에 있습니다."),
        fact("employees", 24, "en" to "Acme employed 18,400 people at year end.",
            "zh" to "截至年底，Acme 员工人数为18400人。",
            "fr" to "Acme employait 18 400 personnes à la fin de l'année.",
            "de" to "Acme beschäftigte zum Jahresende 18.400 Mitarbeiter.",
            "it" to "Acme impiegava 18.400 persone a fine anno.",
            "ja" to "期末時点のAcme従業員数は1万8400人でした。",
            "ko" to "연말 기준 Acme 직원 수는 1만 8400명입니다."),
        fact("margin", 31, "en" to "Operating margin was 31 percent, up two points year over year.",
            "zh" to "营业利润率为31%，同比提高两个百分点。",
            "fr" to "La marge opérationnelle était de 31 %, en hausse de deux points.",
            "de" to "Die operative Marge lag bei 31 Prozent, zwei Punkte über dem Vorjahr.",
            "it" to "Il margine operativo era del 31 percento, due punti in più.",
            "ja" to "営業利益率は31％で、前年比2ポイント上昇しました。",
            "ko" to "영업이익률은 31%로 전년 대비 2포인트 상승했습니다."),
        fact("rnd", 42, "en" to "Research and development spending reached 620 million dollars.",
            "zh" to "研发支出达到6.2亿美元。",
            "fr" to "Les dépenses de recherche et développement ont atteint 620 millions de dollars.",
            "de" to "Die Ausgaben für Forschung und Entwicklung erreichten 620 Millionen Dollar.",
            "it" to "La spesa per ricerca e sviluppo ha raggiunto 620 milioni di dollari.",
            "ja" to "研究開発費は6億2000万ドルに達しました。",
            "ko" to "연구개발비는 6억 2000만 달러에 달했습니다."),
        fact("product", 55, "en" to "The flagship product launched this year is Helio 7.",
            "zh" to "今年推出的旗舰产品是 Helio 7。",
            "fr" to "Le produit phare lancé cette année est Helio 7.",
            "de" to "Das in diesem Jahr eingeführte Flaggschiffprodukt ist Helio 7.",
            "it" to "Il prodotto di punta lanciato quest'anno è Helio 7.",
            "ja" to "今年発売された主力製品はHelio 7です。",
            "ko" to "올해 출시된 주력 제품은 Helio 7입니다."),
        fact("markets", 67, "en" to "Primary markets are Japan, Korea and the European Union.",
            "zh" to "主要市场是日本、韩国和欧盟。",
            "fr" to "Les marchés principaux sont le Japon, la Corée et l'Union européenne.",
            "de" to "Die Hauptmärkte sind Japan, Korea und die Europäische Union.",
            "it" to "I mercati principali sono Giappone, Corea e Unione europea.",
            "ja" to "主要市場は日本、韓国、欧州連合です。",
            "ko" to "주요 시장은 일본, 한국, 유럽연합입니다."),
        fact("carbon", 80, "en" to "Carbon emissions fell 28 percent versus the 2019 baseline.",
            "zh" to "碳排放较2019年基线下降28%。",
            "fr" to "Les émissions de carbone ont baissé de 28 % par rapport à 2019.",
            "de" to "Die Kohlenstoffemissionen sanken um 28 Prozent gegenüber 2019.",
            "it" to "Le emissioni di carbonio sono scese del 28 percento rispetto al 2019.",
            "ja" to "炭素排出量は2019年基準比で28％減少しました。",
            "ko" to "탄소 배출량은 2019년 기준 대비 28% 감소했습니다."),
        fact("dividend", 91, "en" to "The board declared an annual dividend of 1.20 dollars per share.",
            "zh" to "董事会宣布每股年度股息1.20美元。",
            "fr" to "Le conseil a déclaré un dividende annuel de 1,20 dollar par action.",
            "de" to "Der Vorstand beschloss eine Jahresdividende von 1,20 Dollar je Aktie.",
            "it" to "Il consiglio ha dichiarato un dividendo annuale di 1,20 dollari per azione.",
            "ja" to "取締役会は1株当たり1.20ドルの年間配当を決議しました。",
            "ko" to "이사회는 주당 1.20달러의 연간 배당을 결의했습니다."),
        fact("cash", 108, "en" to "Cash and equivalents stood at 2.1 billion dollars.",
            "zh" to "现金及等价物为21亿美元。",
            "fr" to "La trésorerie et équivalents s'élevaient à 2,1 milliards de dollars.",
            "de" to "Zahlungsmittel und Äquivalente betrugen 2,1 Milliarden Dollar.",
            "it" to "Cassa e equivalenti ammontavano a 2,1 miliardi di dollari.",
            "ja" to "現金及び現金同等物は21億ドルでした。",
            "ko" to "현금 및 현금성자산은 21억 달러였습니다."),
        fact("deal", 142, "en" to "Acme completed the acquisition of Northwind Analytics in June.",
            "zh" to "Acme 于6月完成了对 Northwind Analytics 的收购。",
            "fr" to "Acme a finalisé l'acquisition de Northwind Analytics en juin.",
            "de" to "Acme schloss die Übernahme von Northwind Analytics im Juni ab.",
            "it" to "Acme ha completato l'acquisizione di Northwind Analytics a giugno.",
            "ja" to "Acmeは6月にNorthwind Analyticsの買収を完了しました。",
            "ko" to "Acme는 6월에 Northwind Analytics 인수를 완료했습니다."),
        fact("safety", 180, "en" to "There were zero lost-time safety incidents in manufacturing.",
            "zh" to "制造环节损失工时安全事故为零。",
            "fr" to "Il n'y a eu aucun incident de sécurité avec arrêt de travail en production.",
            "de" to "In der Fertigung gab es null Arbeitsunfälle mit Ausfallzeit.",
            "it" to "In produzione non ci sono stati incidenti di sicurezza con perdita di tempo.",
            "ja" to "製造部門の休業災害はゼロ件でした。",
            "ko" to "제조 부문 휴업 재해는 0건이었습니다."),
        fact("patents", 210, "en" to "Patent grants this year totalled 347 across six offices.",
            "zh" to "今年在六个专利局获得347项专利授权。",
            "fr" to "347 brevets ont été délivrés cette année dans six offices.",
            "de" to "In diesem Jahr wurden 347 Patente in sechs Umlaufstellen erteilt.",
            "it" to "Quest'anno sono stati concessi 347 brevetti in sei uffici.",
            "ja" to "今年6つの特許庁で合計347件の特許が成立しました。",
            "ko" to "올해 6개 특허청에서 총 347건의 특허가 등록되었습니다."),
        fact("outlook", 250, "en" to "Management guides 8 to 10 percent revenue growth next year.",
            "zh" to "管理层预计明年营收增长8%至10%。",
            "fr" to "La direction vise une croissance du chiffre d'affaires de 8 à 10 % l'an prochain.",
            "de" to "Das Management erwartet im nächsten Jahr 8 bis 10 Prozent Umsatzwachstum.",
            "it" to "La direzione prevede una crescita dei ricavi dell'8-10 percento l'anno prossimo.",
            "ja" to "経営陣は来期の売上高成長率を8〜10％と見込んでいます。",
            "ko" to "경영진은 내년 매출 성장률을 8~10%로 전망합니다."),
    )

    /** Two phrasings per fact, so each language has 30 questions. */
    private val prompts: Map<String, List<String>> = mapOf(
        "revenue" to listOf("What was full-year revenue in 2024?", "How much revenue did Acme report for 2024?"),
        "ceo" to listOf("Who is the CEO of Acme?", "Who has led Acme since 2019?"),
        "hq" to listOf("Where is Acme headquartered?", "In which city is Acme's headquarters?"),
        "employees" to listOf("How many employees did Acme have?", "What was the year-end headcount?"),
        "margin" to listOf("What was the operating margin?", "How large was operating margin versus last year?"),
        "rnd" to listOf("How much was spent on research and development?", "What was R&D spending?"),
        "product" to listOf("What flagship product launched this year?", "What is Helio 7?"),
        "markets" to listOf("What are Acme's primary markets?", "Which regions are the main markets?"),
        "carbon" to listOf("How much did carbon emissions fall?", "What is the carbon reduction versus 2019?"),
        "dividend" to listOf("What dividend was declared?", "How large is the annual dividend per share?"),
        "cash" to listOf("How much cash did Acme hold?", "What was cash and equivalents?"),
        "deal" to listOf("Which company did Acme acquire?", "When was Northwind Analytics acquired?"),
        "safety" to listOf("How many lost-time safety incidents occurred?", "What was the manufacturing safety record?"),
        "patents" to listOf("How many patents were granted?", "How many patent grants were there this year?"),
        "outlook" to listOf("What revenue growth is guided for next year?", "What is management's growth outlook?"),
    )

    private val promptTranslations: Map<String, Map<String, List<String>>> = mapOf(
        "zh" to mapOf(
            "revenue" to listOf("2024年全年营收是多少？", "Acme 2024年报告了多少收入？"),
            "ceo" to listOf("Acme 的首席执行官是谁？", "谁自2019年起领导 Acme？"),
            "hq" to listOf("Acme 总部位于哪里？", "Acme 总部在哪个城市？"),
            "employees" to listOf("Acme 有多少员工？", "年底员工人数是多少？"),
            "margin" to listOf("营业利润率是多少？", "营业利润率同比如何？"),
            "rnd" to listOf("研发支出是多少？", "研究与开发花了多少钱？"),
            "product" to listOf("今年推出的旗舰产品是什么？", "Helio 7 是什么？"),
            "markets" to listOf("Acme 的主要市场是哪些？", "主要市场覆盖哪些地区？"),
            "carbon" to listOf("碳排放下降了多少？", "相对2019年碳减排是多少？"),
            "dividend" to listOf("宣布了多少股息？", "每股年度股息是多少？"),
            "cash" to listOf("Acme 持有多少现金？", "现金及等价物是多少？"),
            "deal" to listOf("Acme 收购了哪家公司？", "Northwind Analytics 是何时被收购的？"),
            "safety" to listOf("损失工时安全事故有多少？", "制造安全记录如何？"),
            "patents" to listOf("获得了多少项专利授权？", "今年专利授权数量是多少？"),
            "outlook" to listOf("明年营收增长指引是多少？", "管理层的增长展望是什么？"),
        ),
        "fr" to mapOf(
            "revenue" to listOf("Quel était le chiffre d'affaires annuel 2024 ?", "Combien de revenus Acme a-t-elle déclarés en 2024 ?"),
            "ceo" to listOf("Qui est la directrice générale d'Acme ?", "Qui dirige Acme depuis 2019 ?"),
            "hq" to listOf("Où se trouve le siège d'Acme ?", "Dans quelle ville est le siège d'Acme ?"),
            "employees" to listOf("Combien d'employés Acme avait-elle ?", "Quel était l'effectif de fin d'année ?"),
            "margin" to listOf("Quelle était la marge opérationnelle ?", "Comment a évolué la marge opérationnelle ?"),
            "rnd" to listOf("Combien a été dépensé en R&D ?", "Quel était le budget de recherche et développement ?"),
            "product" to listOf("Quel produit phare a été lancé cette année ?", "Qu'est-ce que Helio 7 ?"),
            "markets" to listOf("Quels sont les marchés principaux d'Acme ?", "Quelles régions sont les marchés clés ?"),
            "carbon" to listOf("De combien les émissions de carbone ont-elles baissé ?", "Quelle est la baisse carbone vs 2019 ?"),
            "dividend" to listOf("Quel dividende a été déclaré ?", "Quel est le dividende annuel par action ?"),
            "cash" to listOf("Quelle trésorerie Acme détenait-elle ?", "Combien de cash et équivalents ?"),
            "deal" to listOf("Quelle société Acme a-t-elle acquise ?", "Quand Northwind Analytics a-t-elle été acquise ?"),
            "safety" to listOf("Combien d'incidents de sécurité avec arrêt ?", "Quel est le bilan sécurité en production ?"),
            "patents" to listOf("Combien de brevets ont été délivrés ?", "Combien de brevets cette année ?"),
            "outlook" to listOf("Quelle croissance est prévue l'an prochain ?", "Quelle est la prévision de croissance ?"),
        ),
        "de" to mapOf(
            "revenue" to listOf("Wie hoch war der Jahresumsatz 2024?", "Welchen Umsatz hat Acme 2024 gemeldet?"),
            "ceo" to listOf("Wer ist die Vorstandsvorsitzende von Acme?", "Wer leitet Acme seit 2019?"),
            "hq" to listOf("Wo ist der Hauptsitz von Acme?", "In welcher Stadt liegt der Hauptsitz?"),
            "employees" to listOf("Wie viele Mitarbeiter hatte Acme?", "Wie hoch war der Personalbestand zum Jahresende?"),
            "margin" to listOf("Wie hoch war die operative Marge?", "Wie entwickelte sich die operative Marge?"),
            "rnd" to listOf("Wie hoch waren die F&E-Ausgaben?", "Was wurde für Forschung und Entwicklung ausgegeben?"),
            "product" to listOf("Welches Flaggschiffprodukt wurde dieses Jahr eingeführt?", "Was ist Helio 7?"),
            "markets" to listOf("Was sind Acmes Hauptmärkte?", "Welche Regionen sind die wichtigsten Märkte?"),
            "carbon" to listOf("Um wie viel sanken die Kohlenstoffemissionen?", "Wie hoch ist die CO2-Reduktion gegenüber 2019?"),
            "dividend" to listOf("Welche Dividende wurde beschlossen?", "Wie hoch ist die Jahresdividende je Aktie?"),
            "cash" to listOf("Wie viel Cash hielt Acme?", "Wie hoch waren Zahlungsmittel und Äquivalente?"),
            "deal" to listOf("Welches Unternehmen hat Acme übernommen?", "Wann wurde Northwind Analytics übernommen?"),
            "safety" to listOf("Wie viele Arbeitsunfälle mit Ausfallzeit gab es?", "Wie war die Sicherheitsbilanz in der Fertigung?"),
            "patents" to listOf("Wie viele Patente wurden erteilt?", "Wie viele Patenterteilungen gab es in diesem Jahr?"),
            "outlook" to listOf("Welches Umsatzwachstum wird für nächstes Jahr erwartet?", "Wie lautet der Wachstumsausblick?"),
        ),
        "it" to mapOf(
            "revenue" to listOf("Quali erano i ricavi annui nel 2024?", "Quanti ricavi ha riportato Acme nel 2024?"),
            "ceo" to listOf("Chi è l'amministratrice delegata di Acme?", "Chi guida Acme dal 2019?"),
            "hq" to listOf("Dov'è la sede di Acme?", "In quale città si trova la sede?"),
            "employees" to listOf("Quanti dipendenti aveva Acme?", "Qual era l'organico di fine anno?"),
            "margin" to listOf("Qual era il margine operativo?", "Come è cambiato il margine operativo?"),
            "rnd" to listOf("Quanto è stato speso in R&S?", "Qual era la spesa per ricerca e sviluppo?"),
            "product" to listOf("Quale prodotto di punta è stato lanciato quest'anno?", "Che cos'è Helio 7?"),
            "markets" to listOf("Quali sono i mercati principali di Acme?", "Quali regioni sono i mercati chiave?"),
            "carbon" to listOf("Di quanto sono scese le emissioni di carbonio?", "Qual è la riduzione rispetto al 2019?"),
            "dividend" to listOf("Quale dividendo è stato dichiarato?", "Quanto vale il dividendo annuale per azione?"),
            "cash" to listOf("Quanta cassa deteneva Acme?", "A quanto ammontavano cassa ed equivalenti?"),
            "deal" to listOf("Quale società ha acquisito Acme?", "Quando è stata acquisita Northwind Analytics?"),
            "safety" to listOf("Quanti incidenti di sicurezza con perdita di tempo?", "Qual è il bilancio sicurezza in produzione?"),
            "patents" to listOf("Quanti brevetti sono stati concessi?", "Quante concessioni di brevetto quest'anno?"),
            "outlook" to listOf("Quale crescita dei ricavi è prevista l'anno prossimo?", "Qual è la previsione di crescita?"),
        ),
        "ja" to listOf("revenue", "ceo", "hq", "employees", "margin", "rnd", "product", "markets", "carbon", "dividend", "cash", "deal", "safety", "patents", "outlook")
            .zip(
                listOf(
                    listOf("2024年の通期売上高はいくらですか？", "Acmeの2024年売上はいくらでしたか？"),
                    listOf("AcmeのCEOは誰ですか？", "2019年からAcmeを率いているのは誰ですか？"),
                    listOf("Acmeの本社はどこにありますか？", "本社はどの都市にありますか？"),
                    listOf("Acmeの従業員数は何人ですか？", "期末の人員は何人でしたか？"),
                    listOf("営業利益率は何パーセントですか？", "営業利益率は前年と比べてどうでしたか？"),
                    listOf("研究開発費はいくらですか？", "R&D支出はいくらでしたか？"),
                    listOf("今年発売された主力製品は何ですか？", "Helio 7とは何ですか？"),
                    listOf("Acmeの主要市場はどこですか？", "主な市場はどの地域ですか？"),
                    listOf("炭素排出量はどれだけ減りましたか？", "2019年比の炭素削減はどれくらいですか？"),
                    listOf("配当はいくらですか？", "1株当たりの年間配当はいくらですか？"),
                    listOf("Acmeの現金はいくらですか？", "現金及び現金同等物はいくらでしたか？"),
                    listOf("Acmeが買収した会社はどこですか？", "Northwind Analyticsはいつ買収されましたか？"),
                    listOf("休業災害は何件でしたか？", "製造部門の安全記録はどうでしたか？"),
                    listOf("成立した特許は何件ですか？", "今年の特許成立件数は？"),
                    listOf("来期の売上成長率の見通しは？", "経営陣の成長見通しは何ですか？"),
                ),
            ).toMap(),
        "ko" to listOf("revenue", "ceo", "hq", "employees", "margin", "rnd", "product", "markets", "carbon", "dividend", "cash", "deal", "safety", "patents", "outlook")
            .zip(
                listOf(
                    listOf("2024년 연간 매출은 얼마입니까?", "Acme가 2024년에 보고한 매출은 얼마입니까?"),
                    listOf("Acme의 CEO는 누구입니까?", "2019년부터 Acme를 이끈 사람은 누구입니까?"),
                    listOf("Acme 본사는 어디에 있습니까?", "본사는 어느 도시에 있습니까?"),
                    listOf("Acme 직원은 몇 명입니까?", "연말 인원은 몇 명입니까?"),
                    listOf("영업이익률은 얼마입니까?", "영업이익률은 전년 대비 어떻습니까?"),
                    listOf("연구개발비는 얼마입니까?", "R&D 지출은 얼마였습니까?"),
                    listOf("올해 출시된 주력 제품은 무엇입니까?", "Helio 7은 무엇입니까?"),
                    listOf("Acme의 주요 시장은 어디입니까?", "주요 시장은 어느 지역입니까?"),
                    listOf("탄소 배출량은 얼마나 줄었습니까?", "2019년 대비 탄소 감축은 얼마입니까?"),
                    listOf("배당은 얼마입니까?", "주당 연간 배당은 얼마입니까?"),
                    listOf("Acme의 현금은 얼마입니까?", "현금 및 현금성자산은 얼마였습니까?"),
                    listOf("Acme가 인수한 회사는 어디입니까?", "Northwind Analytics는 언제 인수되었습니까?"),
                    listOf("휴업 재해는 몇 건입니까?", "제조 부문 안전 기록은 어떻습니까?"),
                    listOf("등록된 특허는 몇 건입니까?", "올해 특허 등록 건수는?"),
                    listOf("내년 매출 성장 전망은 무엇입니까?", "경영진의 성장 전망은?"),
                ),
            ).toMap(),
    )

    fun questions(): List<Question> {
        val out = ArrayList<Question>(LANGS.size * facts.size * 2)
        for (lang in LANGS) {
            for (f in facts) {
                val pair = if (lang == "en") prompts.getValue(f.id) else promptTranslations.getValue(lang).getValue(f.id)
                out += Question("${lang}-${f.id}-a", lang, pair[0], setOf(f.page), f.id)
                out += Question("${lang}-${f.id}-b", lang, pair[1], setOf(f.page), f.id)
            }
        }
        return out
    }

    fun chunks(): List<Chunk> {
        val out = ArrayList<Chunk>()
        var idx = 0
        for (f in facts) {
            for (lang in LANGS) {
                out += Chunk(f.id, f.page, lang, f.texts.getValue(lang), idx++)
            }
        }
        return out
    }

    private fun fact(id: String, page: Int, vararg pairs: Pair<String, String>) = Fact(id, page, pairs.toMap())
}
