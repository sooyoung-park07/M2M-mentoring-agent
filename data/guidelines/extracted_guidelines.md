# extracted_guidelines.md

50건 멘티 질문 분석 기반 Agent 1 가이드라인

## 패턴 빈도 요약

| 패턴 | 빈도 | 비율 |
|------|------|------|
| 복합질문(3개+) | 26 | 52% |
| 가능성판단 | 26 | 52% |
| 자료피드백 | 23 | 46% |
| 커리어로드맵 | 21 | 42% |
| 최신트렌드 | 20 | 40% |
| 경험부족형 | 20 | 40% |
| 직무미분화형 | 18 | 36% |
| 직무전환 | 10 | 20% |
| 기술전이 | 7 | 14% |
| 복합질문(2개) | 6 | 12% |
| 멘토경험요청 | 1 | 2% |

---

## Core Guidelines (10건 이상 반복)

### G01. 복합질문 분리 [26건]
멘티가 번호를 붙여 3개 이상의 질문을 묶어 보내는 경우가 전체의 52%다.
이 경우 하나의 refined_question으로 합치지 말고 question_units로 분리한다.
각 unit은 priority(primary/secondary)와 answerability(searchable/mentor_needed)를 반드시 지정한다.
3개 이상이면 scope_too_broad=true로 표시한다.

### G02. 가능성 판단 질문의 위험 처리 [26건]
"이직 가능할까요", "내향인도 될 수 있을까요", "30대에 도전해도 괜찮을까요" 등
합격·이직·적성 가능성을 단정적으로 묻는 질문은 전체의 52%에 달한다.
이런 질문에 AI가 단정적 판단을 내리면 안 된다.
risk_flags에 "success_probability_asked"를 추가하고 mentor_first 전략을 권장한다.

### G03. 자료 직접 피드백 [23건]
"자소서를 어떻게 써야 하나요", "포트폴리오에 어떤 걸 담아야 하나요" 등
실제 문서를 보지 않고는 답할 수 없는 피드백성 질문이 46%다.
자소서/레주메/포트폴리오 직접 검토 요청이 포함된 경우 requires_artifact_review=true.
실제 파일 첨부 여부와 관계없이, 개인 문서 맞춤 피드백 요청이면 설정한다.

### G04. 커리어 로드맵 기대 [21건]
"단기/장기 준비 순서", "마일스톤", "어떤 순서로 준비해야 하나요"가 42%에 등장한다.
expected_answer_type에 "실행로드맵" 또는 "단계별 준비전략"을 포함한다.
단순 정보가 아니라 멘토의 경험 기반 순서가 필요하므로 mentor_first 가중치를 높인다.

### G05. 최신 트렌드·AI 전망 [20건]
"AI 시대 어떻게 살아남나", "향후 직무 전망", "AI가 대체할까요" 등이 40%에 등장한다.
recency_sensitive=true로 표시하고 retrieval 시 recency 가중치를 상향한다.
트렌드 질문이 복합 질문의 일부로 섞여 있을 때도 해당 unit에만 recency_sensitive=true를 붙인다.

### G06. 경험 부족형 [20건]
"인턴 없는데", "대외활동이 부족한데", "경험이 없지만" 등 경험 공백을 언급하는 질문이 40%다.
이는 별도 패턴이 아니라 context 정보다. current_bottleneck에 "경험_공백_극복"을 포함한다.
경험 부족을 개인 문제로 다루지 말고, 어떤 대안 경험이 유효한지로 정제 방향을 잡는다.

### G07. 직무 미분화형 [18건]
"어떤 직무가 맞는지 모르겠다", "방향성을 잡기 어렵다", "막막하다" 등이 36%다.
target_role_specificity=broad 또는 unclear로 설정한다.
이 경우 refined_question 생성 전에 목표 직무를 좁히는 clarification 질문을 먼저 시도한다.
clarification 없이 바로 정제하면 뒤 agent에 잘못된 target_role을 전달하게 된다.

---

## Secondary Guidelines (5~9건 반복)

### G08. 직무 전환형 [10건]
현재 직무에서 다른 직무로 전환하려는 질문이 20%다 (SCM→PM, 은행→증권사, 직무변경 이직 등).
source_role, target_role, bridge_hypothesis를 반드시 추출한다.
bridge_hypothesis는 "현재 직무 경험이 목표 직무의 어떤 역량으로 연결되는가"를 1~2문장으로 쓴다.
단순 이직이 아니라 "경험 재해석"이 핵심임을 current_bottleneck에 반영한다.

### G09. 기술 전이형 [7건]
보유 기술/경험이 다른 분야에서도 통하는지 판단받고 싶어하는 질문이 14%다.
transferable_skills (현재 보유 기술 목록)와 target_domain_candidates (적용 가능 도메인 후보)를 추출한다.
단순 "이직 가능한가" 질문으로 처리하지 않는다. 핵심은 기술의 범용성 판단이다.

---

## Edge Case Guidelines (1~4건, 실패 시 위험)

### G10. 멘토 개인 경험 직접 요청
"멘토님은 어떻게 하셨나요", "선배님의 경험이 궁금합니다" 등 개인 경험을 명시적으로 요청하는 unit은
answerability=mentor_needed로 분리한다. 검색으로 대체 불가.

### G11. 적성/판단형 단일 질문
"내가 이 직무에 맞는지", "이 직무를 계속해야 할지" 등 개인 적성 판단을 요구하는 질문은
어떤 정보도 단정적 답변이 될 수 없다.
risk_flags에 "fitness_judgment_asked" 추가, current_bottleneck에 "진로적합성 불확실"로 표시.

### G12. 자료 첨부 없는 포트폴리오 피드백
포트폴리오 개수/구성 질문은 artifact_review 경계 케이스다.
실제 파일 첨부 없이 일반론적으로 묻는 경우 requires_artifact_review=false이지만,
"제 포트폴리오에서 어떤 점을" 등 개인 자료 언급이 있으면 true로 설정한다.
