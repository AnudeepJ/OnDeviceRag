# safety.pdf QA test bank

Source: `safety.pdf`, a 209-page construction safety manual. Page numbers below are PDF pages.

## Facts and procedures

| # | Question | Expected source-backed answer | Page |
|---|---|---|---:|
| 1 | What are the five basic steps of HIRA? | Initiate HIRA; identify hazards; identify affected parties and effects; assess risk; monitor and review. | 25 |
| 2 | What is the hierarchy for controlling a hazard? | Control at the source (engineering control), then along the path to the worker (administrative control), and PPE only as the last level. | 30 |
| 3 | When must employers provide PPE? | When engineering, work-practice, and administrative controls are not feasible or do not provide sufficient protection. | 163 |
| 4 | What makes a space a confined space? | It can be entered bodily and may have engulfment material, a hazardous atmosphere, inward-converging walls, limited entry/egress, or is not designed for occupancy. | 97 |
| 5 | What is an Emergency Action Plan? | A written emergency-preparedness document with agreed, recorded, and rehearsed response strategies. | 196 |
| 6 | What emergencies must a construction-site plan consider? | Serious injury/poisoning, fire/explosion, chemical leak/spill, electrocution, structural collapse, and flood/earthquake. | 196 |
| 7 | What are the four elements of fire? | Heat/energy, oxygen, fuel, and an exothermic chain reaction. | 175 |
| 8 | Which extinguisher is appropriate for a small electrical fire? | A Class C or multipurpose ABC extinguisher. | 66 |
| 9 | What is a trench? | An excavation whose depth exceeds its width. | 102 |
| 10 | What must be checked before excavation begins? | Soil type, buried services, overhead lines, hazardous atmosphere/oxygen, nearby hazards, water, PPE, access/egress, and emergency arrangements. | 102-103 |

## Numeric and structured requirements

| # | Question | Expected source-backed answer | Page |
|---|---|---|---:|
| 11 | At what trench depth is a protective system generally required? | 1.2 m (4 ft) or deeper, unless excavated entirely in stable rock. | 103 |
| 12 | How far may a worker be from an exit inside a trench? | Usually no more than 8 m (25 ft). | 103 |
| 13 | What is the maximum allowable slope for Type B soil? | 1:1 for simple and benched excavations up to 6 m (20 ft), as specified. | 105 |
| 14 | What minimum load must a scaffold support? | At least four times the anticipated weight of people and material. | 78 |
| 15 | When is a scaffold required to be secured against tipping? | When its height exceeds four times its minimum base dimension. | 78 |
| 16 | At what interval is a scaffold landing platform required? | Every 9 m of height. | 78 |
| 17 | What are the tube-and-coupler scaffold rail and toe-board dimensions? | Mid rail 600 mm, top rail 1200 mm, and toe board 150 mm. | 78 |
| 18 | What must be done before moving a mobile scaffold? | Remove all material and equipment from its platform; ensure sufficient help and watch for holes/overhead obstructions. | 79 |
| 19 | What pressure can high-pressure gas cylinders withstand? | Up to 10,000 psi. | 152 |
| 20 | Which Indian Standard covers safety for excavation work? | IS 3764:1992, Code of safety for excavation work. | 206 |

## Coverage and summary prompts

| # | Question | Expected source-backed answer | Pages |
|---|---|---|---:|
| 21 | Summarize the document. | A construction-safety manual covering policy, site management, incident investigation, HIRA, permits, training, material handling, electrical safety, height work, confined spaces, excavation, equipment, PPE, fire, first aid, occupational health, emergency action, legal compliance, and applicable Indian Standards. | 2, 3-209 |
| 22 | Summarize the work-at-height chapter. | Guardrail, fall-arrest, net, restraint, and scaffold requirements; trained/qualified people must erect or modify scaffolds. | 75-79 |
| 23 | Summarize the excavation chapter. | Defines excavations/trenches, identifies collapse and utility hazards, requires assessment and protective systems, and explains sloping, benching, shoring, and shields. | 102-107 |
| 24 | Summarize the PPE chapter. | Applies the hierarchy of controls, defines employer/employee responsibilities, and covers eye/face and other PPE selection, use, care, and replacement. | 163-174 |

## Initial live run on Nothing A001

`safety.pdf` indexed successfully: 209 pages, 1,427 chunks, Latin script, index v21. Gemma ran on GPU.

| Question | Result |
|---|---|
| Minimum scaffold capacity | Pass: four times, Page 78. |
| Protective-system trench depth | Pass: 1.2 m / 4 ft, Page 103. |
| Trench exit distance | Retrieval pass but answer failure: supported `8` was replaced with `[unverified value]`. |
| Type B soil slope | Main value `1:1` passed, but supporting `6 m` was replaced with `[unverified value]`. |
| Confined-space definition | Retrieval pass but answer failure: returned a generic chapter-navigation response instead of the definition. |
| Four elements of fire | Pass: fuel, energy, oxygen, and chain reaction, Page 175. |
| HIRA steps | Retrieval pass but answer failure: returned only the first of five steps. |
| PPE condition | Pass: required where higher-level controls are infeasible/insufficient, Page 163. |
| Emergency Action Plan | Pass: written, agreed, recorded, rehearsed response plan, Page 196. |
| Scaffold rail/toe-board dimensions | Partial: 600 mm and 1200 mm passed; supported 150 mm was replaced with `[unverified value]`. |
| Excavation Indian Standard | Partial: `IS 3764` passed; its year `1992` was replaced with `[unverified value]`. |
| Document overview | Fail: summarized only the Indian Standards appendix (pages 206-208) and emitted `SECTION null`, rather than the whole manual. |

## Expanded acceptance run

The additional run used fresh user-equivalent prompts on the same Nothing A001 device and GPU backend.

| Question | Result |
|---|---|
| What must be done before moving a mobile scaffold? | Pass: remove material/equipment, use sufficient help, and watch for floor holes and overhead obstructions; Page 79. |
| Who may erect, dismantle, move, or modify scaffolding? | Pass: a qualified person and specifically trained employees, with responsible supervision; Page 77. |
| What pressure can high-pressure gas cylinders withstand? | Pass: up to 10,000 psi; Page 152. The whitespace-separated value/unit form survived grounding. |
| Who is the CEO of the company? | Pass: safe refusal; no unsupported CEO was invented. |
| Summarize the chapter on excavation. | Fail: refused despite the dedicated Chapter 13 content. |
| Summarize Chapter 13 Excavation. | Fail: the explicit chapter number did not resolve the section either. |
| What must happen when a vehicle is reversing on site? | Pass: recommends a one-way system where possible, a signaller, and reversing sirens; Page 24. |
| Type B slope, then “What about Type C soil?” | Follow-up pass: retained excavation context and returned Type C's 1 1/2:1 maximum slope on Page 105. Compact `6Mt` values were still redacted. |
