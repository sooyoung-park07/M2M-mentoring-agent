"""
에이전트 4: 자산화 에이전트 v2

[에이전트 루프]
관찰0: 이중 동의 확인 (멘토 동의 AND 멘티 동의)
관찰1: 최소 길이 확인 (≥ 150자)
관찰2: 중복 체크 (기존 자산과 cosine similarity < 0.85)
행동3: LLM 품질 판단 4종
  - 개인맥락 강도:  isolated / transferable
  - 정보 밀도:     low / high
  - 시의성:       stale / stale_but_useful / timeless  ← v2: 3단계
  - 회사 기밀성:   sensitive / safe
행동4: 메타데이터 생성 (자산화 통과 시)
  - asset_summary, transferability_scope
  - domain/role/question_type/bottleneck/skill/career_stage 태그
관찰4: 최종 저장 (풍부한 record 스키마)

핵심 변경점 (v2):
  - recency 3단계: stale_but_useful은 통과 (requires_latest_check=true 표시)
  - 메타데이터 LLM 생성: Agent 2 검색품질 향상을 위한 태그 전면 확장
  - reject_reasons 구조화: 실패 원인 운영 추적 가능
  - embedding 텍스트: question+answer_summarize → asset_summary+transferability_scope
  - Agent 1 taxonomy_tags 수신: 외부 태그와 병합
"""

import json
import os
from openai import OpenAI
from db.json_db import (
    save_answer, update_session, update_by_id,
    get_assetized_answers, new_id, now_str, DB_DIR,
)
from utils.embedding import get_embedding, top_k_similar

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def merge_unique(*lists) -> list:
    """여러 리스트를 중복 없이 합친다. None/빈 값 무시, 순서 유지."""
    out: list = []
    for lst in lists:
        if not isinstance(lst, list):
            continue
        for x in lst:
            if x and x not in out:
                out.append(x)
    return out


# ─────────────────────────────────────────
# LLM 판단 프롬프트 (품질 gate)
# ─────────────────────────────────────────

PERSONAL_CONTEXT_PROMPT = """이 멘토 답변이 다른 멘티에게 재사용 가능한지 개인 맥락 강도로 판단해줘.

재사용 불가 (isolated) — 하나라도 해당 시:
- 특정 학교명/전공/나이/지역이 3개 이상 조합되어 답변의 핵심 논리를 구성
- 개인 정보를 제거하면 답변의 논리가 무너짐 (예: "부산대 출신이라서 가능한 루트")
- 가족 상황·재정 상황 등 민감 맥락에 근거한 조언
- 특정 멘티에게만 유효한 결론 (예: "선생님 경우엔 OO하는 게 맞아요")

재사용 가능 (transferable):
- 개인 경험에서 출발하지만 일반 원칙/방법론으로 연결됨
- 익명화 후에도 핵심 가치가 유지됨
- 비슷한 상황의 다른 사람에게도 적용 가능

JSON만 출력:
{{"personal_context": "isolated or transferable", "reason": "한 줄"}}

답변:
{answer_content}"""


INFO_DENSITY_PROMPT = """이 멘토 답변의 정보 밀도를 판단해줘.

HIGH 기준 — 하나라도 해당 시:
- 특정 도구/플랫폼/자격증 명시 (프로그래머스, 캐글, SQL, 정보처리기사 등)
- 수치/기간/횟수 포함 (3개월, 주 5회, 연봉 3000만원 등)
- 단계적 방법론이나 순서 있는 조언
- 업계 현실/관행 언급 (공채 vs 수시 비율, 면접 패턴 등)
- 본인 실제 경험에서 나온 구체적 에피소드

LOW 기준:
- 추상적 격려만 ("열심히 하면 돼", "자신감 가져")
- 일반론만 있고 구체적 방법 없음

JSON만 출력:
{{"info_density": "high or low", "reason": "한 줄"}}

답변:
{answer_content}"""


RECENCY_PROMPT = """이 멘토 답변의 시의성을 판단해줘.

stale 기준 (자산화 제외):
- 특정 연도/분기가 답변의 핵심 근거 (예: "2022년 공채에서는")
- 코로나 이전/이후 등 시기 특정 표현이 논리의 중심
- 이미 폐지/변경된 정책·제도 정보

stale_but_useful 기준 (자산화 허용, 최신 확인 필요 표시):
- 내용의 핵심 방법론이나 원칙은 유효하지만 일부 수치·트렌드가 오래됐을 수 있음
- AI, 채용 기준, 연봉 등 변화 가능한 정보가 포함되어 있으나 제거해도 핵심은 남음
- "일반적으로는", "보통" 등 시간 중립적 표현 위주지만 맥락상 1~2년 전 관행 가능성 있음

timeless 기준 (자산화 허용):
- 직무 역량/포트폴리오 방법론 관련 조언
- 경력 개발 원칙과 일반적 취업 전략
- 연도 언급 없이 경험 기반 서술

JSON만 출력:
{{"recency": "stale or stale_but_useful or timeless", "reason": "한 줄"}}

답변:
{answer_content}"""


COMPANY_SENSITIVITY_PROMPT = """이 멘토 답변이 회사 기밀 정보를 포함하는지 판단해줘.

sensitive 기준 — 하나라도 해당 시:
- 특정 회사의 미공개 채용 기준/내부 평가 프로세스
- 연봉 밴드, 승진 기준 등 외부 비공개 인사 정보
- 임직원만 알 수 있는 내부 운영 방식/시스템
- 회사 내부에서만 공유되는 영업·전략 정보

safe 기준:
- 직무 경험/방법론은 본인 경험 기반 서술
- 이미 공개된 채용 공고 수준의 직무 정보
- 일반적으로 알려진 업계 관행

JSON만 출력:
{{"company_sensitivity": "sensitive or safe", "reason": "한 줄"}}

답변:
{answer_content}"""


ASSET_METADATA_PROMPT = """너는 멘토링 답변 자산화 메타데이터 생성기다.
아래 질문과 답변을 보고, 이후 검색·매칭·자산화 관리에 사용할 메타데이터를 생성해라.

[규칙]
- 질문과 답변에 실제로 근거가 있는 태그만 생성한다.
- 개인 식별 정보(이름·학교명·연락처)는 반드시 제거하거나 일반화한다.
- domain_tags와 role_tags를 구분한다.
  - domain_tags: 분야 (예: 금융, IT, 제약/바이오, 반도체)
  - role_tags: 직무 (예: PM/PO, 데이터분석가, QC, SCM)
- question_type_tags는 이 답변이 실제로 다룬 질문 유형만 넣는다.
- bottleneck_tags는 이 답변이 해결하는 병목 유형이다.
- transferability_scope: 너무 넓게 쓰지 말고, 어떤 상황의 멘티에게 유용한지 구체화한다.
- asset_summary는 Agent 2 검색에서 임베딩으로 사용된다. 직무·병목·방법론 키워드 중심으로 작성한다.

[출력 JSON만 반환]
{{
  "domain_tags": [],
  "role_tags": [],
  "question_type_tags": [],
  "bottleneck_tags": [],
  "skill_tags": [],
  "career_stage_tags": [],
  "source_role": "",
  "target_role": "",
  "expected_answer_type": "",
  "transferability_scope": "어떤 상황의 멘티에게 재사용 가능한지 1문장",
  "asset_summary": "검색용 핵심 내용 2~3문장 (직무·병목·방법론 키워드 포함)"
}}

질문:
{question_content}

답변:
{answer_content}"""


# ─────────────────────────────────────────
# 에이전트 클래스
# ─────────────────────────────────────────

class AssetizeAgent:
    SIMILARITY_THRESHOLD = 0.85
    MIN_LENGTH = 150

    def run(
        self,
        session_id: str,
        mentor_id: str,
        question_content: str,
        answer_content: str,
        answer_summarize: str,
        domain_tags: list[str],
        mentor_consent: bool = False,
        mentee_consent: bool = False,
        satisfaction_score: float | None = None,
        # v2 신규 — Agent 1 taxonomy 및 메타데이터
        taxonomy_tags: dict | None = None,
        source_role: str = "",
        target_role: str = "",
        current_bottleneck: str = "",
        expected_answer_type: str = "",
        question_units: list[dict] | None = None,
    ) -> dict:
        print("[자산화 에이전트 v2] 실행 중...")
        taxonomy_tags  = taxonomy_tags or {}
        question_units = question_units or []

        reject_reasons: dict = {
            "consent":             True,
            "length":              True,
            "duplicate":           False,
            "personal_context":    "미판단",
            "info_density":        "미판단",
            "recency":             "미판단",
            "company_sensitivity": "미판단",
        }

        # ── Gate 0: 이중 동의 ──
        print(f"  관찰0 | 멘토 동의: {mentor_consent} / 멘티 동의: {mentee_consent}")
        if not (mentor_consent and mentee_consent):
            missing = []
            if not mentor_consent: missing.append("멘토 미동의")
            if not mentee_consent: missing.append("멘티 미동의")
            reject_reasons["consent"] = False
            reason = ", ".join(missing)
            print(f"  판단0 | 동의 미충족 ({reason}) → 비자산화 저장")
            return self._finalize(
                session_id, mentor_id, question_content, answer_content,
                answer_summarize, domain_tags, satisfaction_score,
                is_assetized=False, reject_reasons=reject_reasons,
            )

        # ── Gate 1: 최소 길이 ──
        length = len(answer_content)
        print(f"  관찰1 | 답변 길이: {length}자 (최소 {self.MIN_LENGTH}자)")
        if length < self.MIN_LENGTH:
            reject_reasons["length"] = False
            print("  판단1 | 길이 미달 → 비자산화 저장")
            return self._finalize(
                session_id, mentor_id, question_content, answer_content,
                answer_summarize, domain_tags, satisfaction_score,
                is_assetized=False, reject_reasons=reject_reasons,
            )

        # ── Gate 2: 중복 체크 ──
        # v2: 임베딩 텍스트를 question+summarize → question+answer_summarize 유지
        # (asset_summary는 아직 없으므로 기존 방식 사용)
        embed_text = f"{question_content}\n{answer_summarize}"
        answer_vec = get_embedding(embed_text)

        existing = get_assetized_answers()
        if existing:
            top = top_k_similar(answer_vec, existing, vec_field="embedding", k=1)
            if top:
                best_match, best_score = top[0]
                print(f"  관찰2 | 최고 유사도: {best_score:.3f} (임계값: {self.SIMILARITY_THRESHOLD})")
                if best_score >= self.SIMILARITY_THRESHOLD:
                    print(f"  판단2 | 중복 감지 ({best_match['answer_id']}) → reuse_count 증가 후 비자산화")
                    increment_reuse(best_match["answer_id"])
                    reject_reasons["duplicate"] = True
                    return self._finalize(
                        session_id, mentor_id, question_content, answer_content,
                        answer_summarize, domain_tags, satisfaction_score,
                        is_assetized=False, reject_reasons=reject_reasons,
                    )
                else:
                    print("  관찰2 | 중복 없음 → LLM 판단 진행")
        else:
            print("  관찰2 | 기존 자산 없음 → LLM 판단 진행")

        # ── Gate 3: LLM 품질 판단 4종 ──
        print("  행동3 | LLM 품질 판단 중...")
        personal    = self._judge(PERSONAL_CONTEXT_PROMPT.format(answer_content=answer_content))
        density     = self._judge(INFO_DENSITY_PROMPT.format(answer_content=answer_content))
        recency     = self._judge(RECENCY_PROMPT.format(answer_content=answer_content))
        sensitivity = self._judge(COMPANY_SENSITIVITY_PROMPT.format(answer_content=answer_content))

        p_val = personal.get("personal_context", "")
        d_val = density.get("info_density", "")
        r_val = recency.get("recency", "")
        s_val = sensitivity.get("company_sensitivity", "")

        p_ok = p_val == "transferable"
        d_ok = d_val == "high"
        r_ok = r_val in ("timeless", "stale_but_useful")   # v2: stale_but_useful도 통과
        s_ok = s_val == "safe"

        reject_reasons["personal_context"]    = p_val
        reject_reasons["info_density"]        = d_val
        reject_reasons["recency"]             = r_val
        reject_reasons["company_sensitivity"] = s_val

        print(
            f"  관찰3 | 개인맥락: {p_val} / 정보밀도: {d_val} / "
            f"시의성: {r_val} / 회사기밀: {s_val}"
        )

        is_assetized = p_ok and d_ok and r_ok and s_ok

        if not is_assetized:
            fails = []
            if not p_ok: fails.append(f"개인맥락({personal.get('reason', '')})")
            if not d_ok: fails.append(f"정보밀도({density.get('reason', '')})")
            if not r_ok: fails.append(f"시의성({recency.get('reason', '')})")
            if not s_ok: fails.append(f"회사기밀({sensitivity.get('reason', '')})")
            print(f"  판단3 | 자산화 미달 → 비자산화 저장 [{' / '.join(fails)}]")
            return self._finalize(
                session_id, mentor_id, question_content, answer_content,
                answer_summarize, domain_tags, satisfaction_score,
                is_assetized=False, reject_reasons=reject_reasons,
            )

        # ── Gate 4 (v2 신규): 메타데이터 생성 ──
        print("  행동4 | 자산 메타데이터 생성 중...")
        metadata = self._generate_metadata(question_content, answer_content)

        # Agent 1 taxonomy_tags + LLM 생성 metadata 병합 (merge_unique 사용)
        final_domain_tags       = merge_unique(domain_tags,        taxonomy_tags.get("domain_tags", []),        metadata.get("domain_tags", []))
        final_role_tags         = merge_unique(                     taxonomy_tags.get("role_tags", []),          metadata.get("role_tags", []))
        final_question_type_tags= merge_unique(                     taxonomy_tags.get("question_type_tags", []), metadata.get("question_type_tags", []))
        final_bottleneck_tags   = merge_unique(                     taxonomy_tags.get("bottleneck_tags", []),    [current_bottleneck] if current_bottleneck else [], metadata.get("bottleneck_tags", []))
        final_skill_tags        = merge_unique(                     taxonomy_tags.get("skill_tags", []),         metadata.get("skill_tags", []))
        final_career_stage_tags = merge_unique(                     taxonomy_tags.get("career_stage_tags", []),  metadata.get("career_stage_tags", []))

        metadata["domain_tags"]        = final_domain_tags
        metadata["role_tags"]          = final_role_tags
        metadata["question_type_tags"] = final_question_type_tags
        metadata["bottleneck_tags"]    = final_bottleneck_tags
        metadata["skill_tags"]         = final_skill_tags
        metadata["career_stage_tags"]  = final_career_stage_tags

        # Agent 1 구조화 필드로 메타데이터 보강 (비어있으면 외부 값 사용)
        if not metadata.get("source_role") and source_role:
            metadata["source_role"] = source_role
        if not metadata.get("target_role") and target_role:
            metadata["target_role"] = target_role
        if not metadata.get("expected_answer_type") and expected_answer_type:
            metadata["expected_answer_type"] = expected_answer_type

        # 최종 임베딩: asset_summary 기반 + 구조화 정보 포함 (검색 특화)
        asset_summary         = metadata.get("asset_summary", "")
        transferability_scope = metadata.get("transferability_scope", "")
        if asset_summary:
            final_embed_text = (
                f"{asset_summary}\n"
                f"재사용 범위: {transferability_scope}\n"
                f"질문유형: {final_question_type_tags}\n"
                f"병목: {final_bottleneck_tags}\n"
                f"직무: {final_role_tags}\n"
                f"기술: {final_skill_tags}"
            )
            answer_vec_final = get_embedding(final_embed_text)
            print(f"  행동4 | asset_summary 기반 구조화 임베딩 생성")
        else:
            answer_vec_final = answer_vec

        requires_latest_check = (r_val == "stale_but_useful")
        if requires_latest_check:
            print("  행동4 | stale_but_useful → requires_latest_check=True 표시")

        print("  판단4 | 자산화 통과 → 자산 DB에 저장")
        return self._finalize(
            session_id, mentor_id, question_content, answer_content,
            answer_summarize, domain_tags, satisfaction_score,
            is_assetized=True,
            embedding=answer_vec_final,
            reject_reasons=reject_reasons,
            metadata=metadata,
            requires_latest_check=requires_latest_check,
            recency_status=r_val,
            question_units=question_units,
        )

    def _judge(self, prompt: str) -> dict:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  _judge 실패: {e}")
            return {}

    def _generate_metadata(self, question_content: str, answer_content: str) -> dict:
        try:
            prompt = ASSET_METADATA_PROMPT.format(
                question_content=question_content,
                answer_content=answer_content[:1500],   # 토큰 절약
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  _generate_metadata 실패: {e} → 빈 메타데이터")
            return {}

    def _finalize(
        self,
        session_id: str,
        mentor_id: str,
        question_content: str,
        answer_content: str,
        answer_summarize: str,
        domain_tags: list[str],
        satisfaction_score: float | None,
        is_assetized: bool,
        reject_reasons: dict | None = None,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
        requires_latest_check: bool = False,
        recency_status: str = "미판단",
        question_units: list[dict] | None = None,
    ) -> dict:
        reject_reasons = reject_reasons or {}
        metadata       = metadata or {}
        question_units = question_units or []

        # reject_reason 문자열 (하위 호환)
        failed = [k for k, v in reject_reasons.items() if v is False or v in ("isolated", "low", "stale", "sensitive")]
        reject_reason_str = " / ".join(failed) if not is_assetized else None

        record = {
            "answer_id":   new_id("ans_"),
            "session_id":  session_id,
            "mentor_id":   mentor_id,

            "question_content": question_content,
            "answer_content":   answer_content,
            "answer_summarize": answer_summarize,
            "asset_summary":    metadata.get("asset_summary", ""),

            # 태그 (merge_unique 병합 결과가 metadata에 이미 저장됨)
            "domain_tags":        metadata.get("domain_tags",        domain_tags),
            "role_tags":          metadata.get("role_tags",          []),
            "question_type_tags": metadata.get("question_type_tags", []),
            "bottleneck_tags":    metadata.get("bottleneck_tags",    []),
            "skill_tags":         metadata.get("skill_tags",         []),
            "career_stage_tags":  metadata.get("career_stage_tags",  []),

            # 직무전환 메타
            "source_role":           metadata.get("source_role",           ""),
            "target_role":           metadata.get("target_role",           ""),
            "expected_answer_type":  metadata.get("expected_answer_type",  ""),
            "transferability_scope": metadata.get("transferability_scope", ""),

            # 세부 질문 구조 (Agent 1)
            "question_units": question_units,

            # 품질 판단 결과
            "recency_status":        recency_status,
            "requires_latest_check": requires_latest_check,
            "company_sensitivity":   reject_reasons.get("company_sensitivity", "미판단"),
            "personal_context":      reject_reasons.get("personal_context",    "미판단"),
            "info_density":          reject_reasons.get("info_density",        "미판단"),

            # 임베딩 및 자산화 여부
            "embedding":          embedding,
            "is_assetized":       is_assetized,
            "reuse_count":        0,
            "satisfaction_score": satisfaction_score,

            # 거절 사유
            "reject_reason":  reject_reason_str,
            "reject_reasons": reject_reasons if not is_assetized else {},
            "created_at":     now_str(),
        }
        save_answer(record)
        label = "✓ 자산화" if is_assetized else "✗ 비자산화"
        print(f"  {label} 저장 완료: {record['answer_id']}")

        update_session(session_id, {
            "answer_id":   record["answer_id"],
            "answer_status": "closed",
            "closed_at":   now_str(),
        })
        print(f"  세션 종료: {session_id}")
        return record


# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────

def update_satisfaction(answer_id: str, score: float) -> bool:
    return update_by_id("mentor_answers.json", "answer_id", answer_id, {
        "satisfaction_score": score,
    })


def increment_reuse(answer_id: str) -> bool:
    path = DB_DIR / "mentor_answers.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for r in data["records"]:
        if r.get("answer_id") == answer_id:
            r["reuse_count"] = r.get("reuse_count", 0) + 1
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
    return False
