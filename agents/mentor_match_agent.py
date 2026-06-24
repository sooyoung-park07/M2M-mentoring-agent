"""
에이전트 3: 멘토 매칭 에이전트

[에이전트 루프]
관찰1: 임베딩 유사도 계산 → 규칙 필터 → 재정렬 → top-K 후보 확보
판단1: 후보 ≥ 3명 AND 평균 유사도 ≥ 0.4?
  충분 → LLM 최종 Top-3 선별
  부족 → 임계값 완화 (0.30 → 0.15) 후 재시도
  끝까지 부족 → 구조화된 실패 반환

Step1: 3채널 임베딩 (프로필×0.30 + 경력×0.30 + 기존답변×0.40)
Step2: 규칙 필터 (최소 유사도 + 수용가능 여부 + 도메인/병목/전환/desired_help 보너스, 총 cap 0.25)
Step3: 재정렬 (유사도×0.55 + 만족도×0.20 + 활동량×0.10 + 수용가능×0.15)
Step4: LLM Top-3 선별 (7개 기준, 후보 외 mentor_id 출력 시 제거)
"""

import os
import json
from openai import OpenAI
from db.json_db import (
    get_all_mentors, get_mentor_experiences, get_all_answers,
    update_session,
)
from utils.embedding import (
    get_embedding, cosine_similarity,
    build_profile_text, build_career_text, build_answer_text,
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


RERANK_PROMPT = """너는 맨투맨 진로 멘토링 서비스의 멘토 매칭 에이전트야.
아래 멘티 질문과 구조화 프로파일, 후보 멘토 정보를 바탕으로 가장 적합한 멘토를 최대 3명 선정해.

[중요 원칙]
- 반드시 후보 목록에 있는 mentor_id 중에서만 선택한다. 후보에 없는 mentor_id나 이름을 생성하지 마.
- 후보 정보에 없는 경력, 성향, 전문성, 상담 경험은 추측하지 않는다.
- 종합 점수는 1차 알고리즘이 계산한 정량 점수이므로 기본적으로 존중한다.
- 다만 멘티 구조화 프로파일과 명백히 더 잘 맞는 후보가 있으면 순위를 조정할 수 있다.
- 순위를 조정한 경우 recommendation_reason에 그 이유를 설명한다.
- 적합한 후보가 3명 미만이면 억지로 3명을 채우지 말고 가능한 후보만 반환한다.
- 추천 이유는 멘티가 이해할 수 있도록 후보 정보에 있는 내용만 근거로 쓴다.
- 개인 식별 가능하거나 민감한 정보는 추천 이유에 포함하지 않는다.
- mentor_id, 이름, 현재 직무는 후보 정보에 있는 값만 사용한다.

[멘티 구조화 프로파일]
- 현재 역할(source_role): {source_role}
- 목표 역할(target_role): {target_role}
- 현재 병목(current_bottleneck): {current_bottleneck}
- 기대 답변 유형(expected_answer_type): {expected_answer_type}
- 전환 연결 가설(bridge_hypothesis): {bridge_hypothesis}
- 필요 멘토 역량(needed_expertise): {needed_expertise}
- 세부 질문 단위(question_units): {question_units}

[평가 기준]
1. job_match: 멘티 목표 직무(target_role) vs 멘토 현재 직무 일치도
2. career_path_match: 멘티 진로 단계/전환 상황 vs 멘토 경력 경로 유사도
3. domain_match: 멘티 관심 도메인 vs 멘토 전문 분야 일치도
4. similar_question_exp: 멘토 기존 답변 요약에 유사 질문 대응 경험이 있는가
5. availability_fit: 후보 정보 기준으로 현재 질문 수용 가능해 보이는가
   (정보가 없으면 "정보 부족"으로 표기, 추측하지 마)
6. bottleneck_fit: 멘티의 current_bottleneck을 해결할 수 있는 경험이 멘토에게 있는가
7. transition_bridge_fit: source_role → target_role 전환의 연결 논리를 도와줄 수 있는가
   (전환이 없으면 N/A)

[점수 표현] 상 / 중 / 하 / 정보 부족 / N/A

[출력 형식] JSON만 출력해.
{{
  "top3": [
    {{
      "mentor_id": "후보 목록에 있는 mentor_id",
      "name": "후보 목록에 있는 이름",
      "rank": 1,
      "recommendation_reason": "멘티 병목·전환 가설·기대 답변 유형을 근거로 한 추천 이유 2~3문장",
      "match_scores": {{
        "job_match": "상/중/하/정보 부족",
        "career_path_match": "상/중/하/정보 부족",
        "domain_match": "상/중/하/정보 부족",
        "similar_question_exp": "상/중/하/정보 부족",
        "availability_fit": "상/중/하/정보 부족",
        "bottleneck_fit": "상/중/하/정보 부족",
        "transition_bridge_fit": "상/중/하/N/A"
      }},
      "caution": "부족하거나 확인이 필요한 점 한 문장 (없으면 없음)"
    }}
  ],
  "selection_summary": "전체적으로 어떤 기준으로 선별했는지 한 문장"
}}

멘티 정제 질문: {question}
멘티 맥락: {context}

후보 멘토들:
{candidates}"""


class MentorMatchAgent:
    MIN_SIM_INITIAL  = 0.3
    MIN_SIM_RELAXED  = 0.15
    QUALITY_MIN_CANDIDATES = 3
    QUALITY_MIN_AVG_SIM    = 0.4

    # 재정렬 가중치
    W_SIM          = 0.55
    W_SATISFACTION = 0.20
    W_ACTIVITY     = 0.10
    W_AVAILABILITY = 0.15

    # 3채널 임베딩 가중치
    W_PROFILE = 0.30
    W_CAREER  = 0.30
    W_ANSWERS = 0.40

    def __init__(self):
        self._cache_profile: dict[str, list[float]] = {}
        self._cache_career:  dict[str, list[float]] = {}
        self._cache_answers: dict[str, list[float]] = {}

    def run(
        self,
        session_id: str,
        refined_question: str,
        conversation_summary: str,
        mentee_constraints: dict | None = None,
    ) -> dict:
        print("[멘토 매칭 에이전트] 실행 중...")

        mentee_constraints = mentee_constraints or {}

        all_mentors    = get_all_mentors()
        active_mentors = [m for m in all_mentors if m.get("active", True)]
        print(f"  활성 멘토: {len(active_mentors)}명")

        if not active_mentors:
            return self._empty_result("활성 멘토 없음", mentee_constraints)

        all_answers = get_all_answers()

        effective_query = mentee_constraints.get("match_query", refined_question)
        if effective_query != refined_question:
            print("  match_query 사용 (Agent 1 최적화 쿼리)")
        query_vec = get_embedding(effective_query)

        # ── 에이전트 루프: 최대 2회 시도 ──
        min_sim = self.MIN_SIM_INITIAL
        top_k   = []
        last_reason = ""

        for attempt in range(1, 3):
            print(f"  [시도 {attempt}] 최소 유사도 임계값: {min_sim}")

            scored   = self._score_by_embedding(query_vec, active_mentors, all_answers)
            filtered = self._rule_filter(scored, mentee_constraints, min_sim=min_sim, all_answers=all_answers)
            reranked = self._rerank_by_feedback(filtered, all_answers)
            top_k    = reranked[:10]

            quality_ok, last_reason = self._evaluate_quality(top_k)
            print(f"  관찰{attempt} | 후보 {len(top_k)}명 | {'충분' if quality_ok else '부족'} ({last_reason})")

            if quality_ok:
                break

            if attempt < 2:
                print(f"  판단{attempt} | 기준 완화 후 재검색 ({min_sim} → {self.MIN_SIM_RELAXED})")
                min_sim = self.MIN_SIM_RELAXED
            else:
                print(f"  판단{attempt} | 재시도 후에도 부족 → 적합한 멘토 없음")
                return self._empty_result(last_reason, mentee_constraints)

        result = self._llm_select_top3(
            refined_question, conversation_summary, top_k, all_answers, mentee_constraints
        )

        if result["top3"]:
            update_session(session_id, {
                "mentor_id": result["top3"][0]["mentor_id"],
                "answer_status": "mentor_matched",
            })

        return result

    # ── 내부 메서드 ──

    def _empty_result(self, reason: str, mentee_constraints: dict) -> dict:
        """구조화된 실패 반환 — 어떤 멘토 풀이 부족한지 추적 가능"""
        return {
            "top3": [],
            "total_candidates": 0,
            "match_status": "no_suitable_mentor",
            "match_failure_reason": reason,
            "needed_mentor_profile": mentee_constraints.get("mentor_match_hints", {}),
        }

    def _evaluate_quality(self, top_k: list[tuple[dict, float]]) -> tuple[bool, str]:
        if len(top_k) < self.QUALITY_MIN_CANDIDATES:
            return False, f"후보 {len(top_k)}명 < 최소 {self.QUALITY_MIN_CANDIDATES}명"
        avg_sim = sum(score for _, score in top_k) / len(top_k)
        if avg_sim < self.QUALITY_MIN_AVG_SIM:
            return False, f"평균 유사도 {avg_sim:.3f} < {self.QUALITY_MIN_AVG_SIM}"
        return True, f"후보 {len(top_k)}명, 평균 유사도 {avg_sim:.3f}"

    def _score_by_embedding(
        self,
        query_vec: list[float],
        mentors: list[dict],
        all_answers: list[dict],
    ) -> list[tuple[dict, float]]:
        """3채널 임베딩 유사도 계산 (프로필·경력·기존답변)"""
        scored = []
        for mentor in mentors:
            mentor_id   = mentor["mentor_id"]
            experiences = get_mentor_experiences(mentor_id)
            answer_sums = [
                a["answer_summarize"]
                for a in all_answers
                if a["mentor_id"] == mentor_id and a.get("answer_summarize")
            ]

            if mentor_id not in self._cache_profile:
                self._cache_profile[mentor_id] = get_embedding(build_profile_text(mentor))
            sim_profile = cosine_similarity(query_vec, self._cache_profile[mentor_id])

            if mentor_id not in self._cache_career:
                career_text = build_career_text(experiences)
                self._cache_career[mentor_id] = (
                    get_embedding(career_text) if career_text.strip()
                    else self._cache_profile[mentor_id]
                )
            sim_career = cosine_similarity(query_vec, self._cache_career[mentor_id])

            if mentor_id not in self._cache_answers:
                answer_text = build_answer_text(answer_sums)
                self._cache_answers[mentor_id] = (
                    get_embedding(answer_text) if answer_text.strip()
                    else self._cache_profile[mentor_id]
                )
            sim_answers = cosine_similarity(query_vec, self._cache_answers[mentor_id])

            sim = (
                sim_profile * self.W_PROFILE
                + sim_career  * self.W_CAREER
                + sim_answers * self.W_ANSWERS
            )
            print(
                f"    {mentor_id} | 프로필={sim_profile:.3f} 경력={sim_career:.3f}"
                f" 답변={sim_answers:.3f} → 통합={sim:.3f}"
            )
            scored.append((mentor, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rule_filter(
        self,
        scored_mentors: list[tuple[dict, float]],
        constraints: dict,
        min_sim: float = 0.3,
        all_answers: list[dict] | None = None,
    ) -> list[tuple[dict, float]]:
        """
        1차 규칙 필터: 최소 유사도 + 수용 가능 여부 + 다중 보너스
        보너스 구조 (총 cap 0.25):
          도메인 태그 일치                   +0.05/개, 최대 0.10
          target_role 직무 매칭              +0.05
          transition_type 전환 경험          +0.05
          desired_help 답변 경험             +0.03
          current_bottleneck 매핑 키워드     +0.05 (스펙 기준 정밀화)
          bridge_hypothesis 키워드           +0.05
          transferable_skills / target_domain +0.05 ("보유기술_전이가능성_판단")
          needed_expertise 항목당            +0.02, 최대 0.06
        """
        interest_domain         = constraints.get("interest_domain", [])
        target_role             = constraints.get("target_role", "")
        transition_type         = constraints.get("transition_type", "미상")
        desired_help            = constraints.get("desired_help", "")
        current_bottleneck      = constraints.get("current_bottleneck", "")
        bridge_hypothesis       = constraints.get("bridge_hypothesis", "")
        transferable_skills     = constraints.get("transferable_skills", [])
        target_domain_candidates = constraints.get("target_domain_candidates", [])
        needed_expertise        = constraints.get("needed_expertise", [])
        all_answers             = all_answers or []

        TRANSITION_KEYWORDS = ["전환", "전직", "이직", "비전공", "전공 변경", "커리어 체인지"]
        HELP_KEYWORDS = {
            "스킬 준비":    ["스킬", "기술", "공부", "자격증", "로드맵"],
            "직무 이해":    ["직무", "현직", "실무", "업무"],
            "포트폴리오":   ["포트폴리오", "프로젝트", "작업물"],
            "면접":         ["면접", "인터뷰", "자소서"],
            "진로 적합성":  ["적합", "맞는", "어울리는", "성향"],
            "서류 전략":    ["서류", "이력서", "레주메", "자소서", "포지셔닝"],
            "커리어 로드맵": ["로드맵", "커리어", "장기", "방향"],
        }
        # 스펙 기준 bottleneck 매핑 (정밀화)
        BOTTLENECK_BONUS = {
            "기존경력_재해석":           (["전환", "이직", "기획", "포지셔닝", "경력기술서", "자소서"], 0.05),
            "전환논리_부족":             (["전환", "이직", "기획", "포지셔닝", "경력기술서", "자소서"], 0.05),
            "자료피드백_필요":           (["자소서", "포트폴리오", "레주메", "면접", "피드백"], 0.05),
            "직무_미분화":               (["직무 비교", "커리어 상담", "진로", "직무 탐색", "직무 이해"], 0.03),
            "경험_공백_극복":            (["공백", "비전공", "인턴", "대외활동", "경험 부족"], 0.05),
            "가능성_불확실":             (["이직", "전환", "도전", "가능성"], 0.03),
            "최신정보_부족":             (["트렌드", "AI", "변화", "전망", "최신"], 0.03),
            "실행순서_불명확":           (["로드맵", "순서", "준비", "우선순위"], 0.03),
            "보유기술_전이가능성_판단":   ([], 0.0),  # 별도 로직으로 처리
        }

        filtered = []
        for mentor, score in scored_mentors:
            if score < min_sim:
                continue
            if not mentor.get("accepting_new_questions", True):
                print(f"  필터 제외: {mentor['mentor_id']} (accepting_new_questions=False)")
                continue

            # mentor_text: 프로필 + 현재 역할 + 기존 답변 요약 합산
            mentor_id = mentor["mentor_id"]
            answer_summaries = [
                a.get("answer_summarize", "")
                for a in all_answers
                if a.get("mentor_id") == mentor_id and a.get("answer_summarize")
            ]
            mentor_text = " ".join([
                mentor.get("matching_summary_text", ""),
                mentor.get("current_role", ""),
                " ".join(answer_summaries),
            ]).lower()

            bonus = 0.0

            # 1. 도메인 태그 보너스 (최대 0.10)
            if interest_domain:
                matching = [d for d in interest_domain if d.lower() in mentor_text]
                bonus += min(len(matching) * 0.05, 0.10)

            # 2. target_role 직무 매칭 보너스
            if target_role and target_role.lower() in mentor_text:
                bonus += 0.05

            # 3. 전환 경험 보너스
            if transition_type not in ("없음", "미상", ""):
                if any(kw in mentor_text for kw in TRANSITION_KEYWORDS):
                    bonus += 0.05

            # 4. desired_help 키워드 보너스
            if desired_help and desired_help in HELP_KEYWORDS:
                if any(kw in mentor_text for kw in HELP_KEYWORDS[desired_help]):
                    bonus += 0.03

            # 5. current_bottleneck 보너스 (스펙 기준 정밀화)
            if current_bottleneck:
                if current_bottleneck == "보유기술_전이가능성_판단":
                    # transferable_skills 또는 target_domain_candidates 매칭
                    all_transfer_kw = list(transferable_skills) + list(target_domain_candidates)
                    if any(kw.lower() in mentor_text for kw in all_transfer_kw if kw):
                        bonus += 0.05
                else:
                    kws, b = BOTTLENECK_BONUS.get(current_bottleneck, ([], 0.0))
                    if kws and b and any(kw in mentor_text for kw in kws):
                        bonus += b

            # 6. bridge_hypothesis 키워드 보너스
            if bridge_hypothesis:
                bridge_words = [w for w in bridge_hypothesis.split() if len(w) > 1][:8]
                if any(w in mentor_text for w in bridge_words):
                    bonus += 0.05

            # 7. needed_expertise 항목당 +0.02 (최대 0.06)
            if needed_expertise:
                expertise_matches = sum(
                    1 for exp in needed_expertise
                    if isinstance(exp, str) and exp.lower() in mentor_text
                )
                bonus += min(expertise_matches * 0.02, 0.06)

            bonus = min(bonus, 0.25)   # 총 보너스 캡
            filtered.append((mentor, score + bonus))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    def _rerank_by_feedback(
        self,
        scored: list[tuple[dict, float]],
        all_answers: list[dict],
    ) -> list[tuple[dict, float]]:
        """2차 재정렬: 유사도×0.55 + 만족도×0.20 + 활동량×0.10 + 수용가능×0.15"""
        if not scored:
            return []

        sims = [s for _, s in scored]
        sim_min, sim_max = min(sims), max(sims)
        sim_range = sim_max - sim_min if sim_max > sim_min else 1.0

        reranked = []
        for mentor, sim_score in scored:
            mentor_id      = mentor["mentor_id"]
            mentor_answers = [a for a in all_answers if a["mentor_id"] == mentor_id]

            sat_scores = [
                a["satisfaction_score"]
                for a in mentor_answers
                if a.get("satisfaction_score") is not None
            ]
            satisfaction = sum(sat_scores) / len(sat_scores) if sat_scores else 0.5
            activity     = min(len(mentor_answers) / 10, 1.0)
            availability = 1.0 if mentor.get("accepting_new_questions", True) else 0.0
            sim_norm     = (sim_score - sim_min) / sim_range

            final = (
                sim_norm     * self.W_SIM
                + satisfaction * self.W_SATISFACTION
                + activity     * self.W_ACTIVITY
                + availability * self.W_AVAILABILITY
            )
            reranked.append((mentor, final))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _llm_select_top3(
        self,
        question: str,
        context: str,
        top_k: list[tuple[dict, float]],
        all_answers: list[dict],
        mentee_constraints: dict | None = None,
    ) -> dict:
        """LLM이 Top-K 중 Top-3 선별 + 추천 근거 생성, 출력 후처리로 후보 외 ID 차단"""
        mentee_constraints = mentee_constraints or {}
        valid_ids = {m["mentor_id"] for m, _ in top_k}

        # ── context 보강 ──
        constraint_parts = []
        if mentee_constraints.get("desired_help"):
            constraint_parts.append(f"원하는 도움: {mentee_constraints['desired_help']}")
        if mentee_constraints.get("transition_type") not in ("없음", "미상", "", None):
            constraint_parts.append(f"전환 유형: {mentee_constraints['transition_type']}")
        if mentee_constraints.get("constraints"):
            constraint_parts.append(f"제약 조건: {', '.join(mentee_constraints['constraints'])}")
        if mentee_constraints.get("current_bottleneck"):
            constraint_parts.append(f"현재 병목: {mentee_constraints['current_bottleneck']}")
        if mentee_constraints.get("expected_answer_type"):
            constraint_parts.append(f"기대 답변 유형: {mentee_constraints['expected_answer_type']}")
        if mentee_constraints.get("bridge_hypothesis"):
            constraint_parts.append(f"전환 연결 가설: {mentee_constraints['bridge_hypothesis']}")
        if mentee_constraints.get("source_role"):
            constraint_parts.append(f"현재 역할: {mentee_constraints['source_role']}")
        if mentee_constraints.get("transferable_skills"):
            constraint_parts.append(f"전이 역량: {', '.join(mentee_constraints['transferable_skills'])}")

        if constraint_parts:
            context = context + "\n[멘티 추가 맥락] " + " / ".join(constraint_parts)

        # ── Agent 2 fallback hints 반영 ──
        mentor_match_hints = mentee_constraints.get("mentor_match_hints", {})
        needed_expertise   = mentor_match_hints.get("needed_expertise", [])
        if isinstance(needed_expertise, list):
            needed_expertise_str = ", ".join(needed_expertise) if needed_expertise else "없음"
        else:
            needed_expertise_str = str(needed_expertise)

        question_units = mentee_constraints.get("question_units", [])
        if isinstance(question_units, list):
            qu_str = "; ".join(
                u.get("question", "") for u in question_units
                if isinstance(u, dict) and u.get("question")
            ) or "없음"
        else:
            qu_str = "없음"

        # ── 후보 직렬화 ──
        candidates_text = ""
        for i, (mentor, score) in enumerate(top_k):
            mentor_id = mentor["mentor_id"]
            answer_summaries = [
                a["answer_summarize"]
                for a in all_answers
                if a["mentor_id"] == mentor_id and a.get("answer_summarize")
            ][:3]

            candidates_text += (
                f"\n[후보 {i+1}] mentor_id: {mentor_id}\n"
                f"이름: {mentor['mentor_info']['name']}\n"
                f"현재 직무: {mentor.get('current_role', '')}\n"
                f"경력: {mentor.get('years_of_experience', 0)}년\n"
                f"프로필: {mentor.get('matching_summary_text', '')[:200]}\n"
                f"기존 답변 요약: {' / '.join(answer_summaries) if answer_summaries else '없음'}\n"
                f"종합 점수: {score:.3f}\n"
            )

        prompt = RERANK_PROMPT.format(
            question=question,
            context=context,
            candidates=candidates_text,
            source_role=mentee_constraints.get("source_role", "없음"),
            target_role=mentee_constraints.get("target_role", "없음"),
            current_bottleneck=mentee_constraints.get("current_bottleneck", "없음"),
            expected_answer_type=mentee_constraints.get("expected_answer_type", "없음"),
            bridge_hypothesis=mentee_constraints.get("bridge_hypothesis", "없음"),
            needed_expertise=needed_expertise_str,
            question_units=qu_str,
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  LLM 선별 실패: {e} → 상위 3명 폴백")
            result = {"top3": [], "selection_summary": "LLM 실패로 점수 기준 상위 3명 선택"}
            for i, (mentor, score) in enumerate(top_k[:3]):
                result["top3"].append({
                    "mentor_id": mentor["mentor_id"],
                    "name": mentor["mentor_info"]["name"],
                    "rank": i + 1,
                    "recommendation_reason": f"종합 점수 {score:.3f}로 상위 선정",
                    "match_scores": {},
                    "caution": "LLM 추천 실패로 알고리즘 점수만 반영됨",
                })

        # ── LLM 출력 검증 ──
        raw_top3 = result.get("top3", [])

        valid_top3 = [item for item in raw_top3 if item.get("mentor_id") in valid_ids]
        if len(valid_top3) < len(raw_top3):
            print(f"  출력 검증 | 후보 외 mentor_id {len(raw_top3) - len(valid_top3)}개 제거됨")

        for i, item in enumerate(valid_top3):
            item["rank"] = i + 1

        valid_top3 = valid_top3[:3]

        mentor_dict = {m["mentor_id"]: m for m, _ in top_k}
        for item in valid_top3:
            mentor = mentor_dict.get(item["mentor_id"], {})
            item["mentor_info"]  = mentor.get("mentor_info", {})
            item["current_role"] = mentor.get("current_role", "")

        result["top3"]             = valid_top3
        result["total_candidates"] = len(top_k)
        return result
