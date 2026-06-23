# agent1_prompt_patch.md

50건 분석 기반 Agent 1 REFINEMENT_PROMPT 반영 내용

---

## [Hard Case Handling Rules] 섹션 교체안

```
[Hard Case Handling Rules]

Rule 1. 복합 질문 분리 (scope_too_broad)
멘티가 번호(1, 2, 3 또는 첫째/둘째/셋째)를 붙여 3개 이상의 질문을 묶어 보낸 경우:
- refined_question 하나로 합치지 말고 question_units로 분리한다.
- 각 unit에 priority(primary/secondary), answerability(searchable/mentor_needed/artifact_needed)를 지정한다.
- scope_too_broad=true로 설정한다.
예: "1. 자소서 어떻게 쓰나요 2. 커피챗 질문 준비법 3. 커리어 마일스톤" → 3개 unit 분리

Rule 2. 가능성/성공 판단 질문 처리 (risk_flags)
"이직 가능할까요", "내향인도 될 수 있나요", "30대에 도전해도 되나요" 등
합격·이직·적성 가능성을 단정적으로 묻는 질문:
- risk_flags에 "success_probability_asked" 추가
- search_strategy_hint=mentor_first 설정
- AI가 성공 여부를 단정하지 않도록 expected_answer_type에 "경험 기반 가이던스"로 표시

Rule 3. 자료 직접 피드백 요청 (requires_artifact_review)
"제 자소서/포트폴리오를 어떻게 써야 하나요" 등 개인 문서 맞춤 피드백 요청:
- requires_artifact_review=true 설정
- 실제 파일 첨부 여부와 무관하게, 개인 문서 내용 기반 피드백이면 true
- "포트폴리오 몇 개가 적당한가" 같은 일반론적 질문은 false

Rule 4. 최신 트렌드·AI 전망 질문 (recency_sensitive)
"AI 시대에 어떻게 살아남나", "향후 직무 전망", "AI가 대체하는가" 등:
- recency_sensitive=true 설정
- 복합 질문에서 트렌드 unit만 recency_sensitive=true로 분리 표시
- 2~3년 이상 지난 정보로 답하면 안 되는 질문

Rule 5. 직무 미분화 (target_role_specificity)
"어떤 직무가 나에게 맞는지 모른다", "방향을 못 잡겠다" 등:
- target_role_specificity=broad 또는 unclear
- refined_question 생성 전 목표 직무 범위를 좁히는 정제 방향으로 유도한다
- clarification 없이 정제하면 뒤 agent에 잘못된 target_role이 전달됨

Rule 6. 직무 전환형 (source_role / bridge_hypothesis)
현재 직무에서 다른 직무로 전환하려는 질문 (이직, 직무 전환, 경력 재해석 등):
- source_role: 현재 직무·경력 배경
- target_role: 목표 직무
- bridge_hypothesis: "현재 경험이 목표 직무의 어떤 역량으로 연결되는가" (1~2문장)
- current_bottleneck: "기존 경력을 목표 직무 언어로 재해석하는 방법을 모름"으로 설정
단순 이직 가능성 질문이 아니라 "경험 재해석"이 핵심임을 명심한다.

Rule 7. 기술 전이형 (transferable_skills / target_domain_candidates)
"제 기술/경험이 다른 분야에서도 통하는가" 등 기술의 범용성 판단 질문:
- transferable_skills: 현재 보유 기술 목록 추출
- target_domain_candidates: 해당 기술이 적용될 수 있는 도메인 후보 목록
- "이직 가능한가"로 단순화하지 않는다. 핵심은 기술의 전이 가능성 판단이다.

Rule 8. 경험 부족형 컨텍스트 처리
"인턴이 없는데", "대외활동이 부족한데" 등 경험 공백을 언급하는 경우:
- current_bottleneck에 "경험_공백_극복"을 포함한다
- 경험 부족 자체가 아니라 "어떤 대안 경험이 유효한가"로 정제 방향을 잡는다
- 이 패턴이 있으면 mentor_first 가중치를 높이는 것이 유리하다
```

---

## question_units 스키마 보완

각 unit에 다음 필드 추가:
```json
{
  "unit_id": 1,
  "type": "직무전환|기술전이|경험재해석|실행전략|트렌드판단|자료피드백|멘토경험|적합성판단",
  "question": "구체적 하위 질문",
  "priority": "primary|secondary",
  "answerability": "searchable|mentor_needed|artifact_needed",
  "recency_sensitive": false
}
```

type 값 선택 기준:
- 직무전환: source→target 전환 경로 질문
- 기술전이: 보유 기술의 타 분야 적용 가능성
- 경험재해석: 기존 경력을 새 직무 언어로 변환
- 실행전략: 준비 순서, 로드맵, 단기/장기 액션플랜
- 트렌드판단: AI·채용변화·직무전망 등 최신 정보 기반 판단
- 자료피드백: 자소서/포트폴리오/이력서 직접 개선 요청
- 멘토경험: 멘토 본인 경험 공유 요청
- 적합성판단: "내가 이 직무에 맞는가" 적성/가능성 판단

---

## current_bottleneck 표준 값 목록 (추천)

50건 분석 기반 자주 등장하는 bottleneck:

- "기존경력_재해석" — 현재 경험을 목표 직무 언어로 번역하는 방법을 모름
- "경험_공백_극복" — 인턴/대외활동 등 직무 관련 경험이 없음
- "직무_미분화" — 여러 직무 중 무엇을 목표로 해야 할지 모름
- "자료피드백_필요" — 자소서/포트폴리오 등 실제 문서 개선 방향 불명확
- "가능성_불확실" — 이직/합격/도전 가능성에 대한 확신 부족
- "최신정보_부족" — AI·트렌드·채용시장 변화에 대한 최신 정보 필요
- "기술전이_판단" — 보유 기술이 목표 직무/도메인에서 유효한지 모름
- "실행순서_불명확" — 준비해야 할 것은 알지만 우선순위와 순서를 모름
