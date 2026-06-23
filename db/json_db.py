"""
JSON 파일 기반 DB CRUD 유틸
각 JSON 파일은 { "schema_name": ..., "fields": ..., "records": [...] } 구조
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any

DB_DIR = Path(__file__).resolve().parent.parent / "json_db"


# ─────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────

def _load(filename: str) -> dict:
    path = DB_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(filename: str, data: dict) -> None:
    path = DB_DIR / filename
    content = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────
# 공통 CRUD
# ─────────────────────────────────────────

def get_all(filename: str) -> list[dict]:
    return _load(filename)["records"]


def get_by_id(filename: str, id_field: str, id_value: str) -> dict | None:
    records = get_all(filename)
    for r in records:
        if r.get(id_field) == id_value:
            return r
    return None


def insert(filename: str, record: dict) -> dict:
    data = _load(filename)
    data["records"].append(record)
    _save(filename, data)
    return record


def update_by_id(filename: str, id_field: str, id_value: str, updates: dict) -> bool:
    data = _load(filename)
    for r in data["records"]:
        if r.get(id_field) == id_value:
            r.update(updates)
            _save(filename, data)
            return True
    return False


def upsert(filename: str, id_field: str, record: dict) -> dict:
    """있으면 업데이트, 없으면 삽입"""
    data = _load(filename)
    id_value = record[id_field]
    for i, r in enumerate(data["records"]):
        if r.get(id_field) == id_value:
            data["records"][i].update(record)
            _save(filename, data)
            return data["records"][i]
    data["records"].append(record)
    _save(filename, data)
    return record


# ─────────────────────────────────────────
# 도메인별 헬퍼
# ─────────────────────────────────────────

def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────
# 멘티 (장기 프로필)
# ─────────────────────────────────────────

def get_mentee(mentee_id: str) -> dict | None:
    return get_by_id("mentees.json", "mentee_id", mentee_id)


def save_mentee(record: dict) -> dict:
    return upsert("mentees.json", "mentee_id", record)


def get_mentee_experiences(mentee_id: str) -> list[dict]:
    return [r for r in get_all("mentee_experiences.json") if r["mentee_id"] == mentee_id]


def save_mentee_experience(record: dict) -> dict:
    return upsert("mentee_experiences.json", "experience_id", record)


def update_mentee_persistent_bottleneck(mentee_id: str, bottleneck: str) -> bool:
    """
    persistent_bottlenecks 리스트에 bottleneck을 누적 추가.
    중복은 무시하고 순서는 최근 항목이 앞으로 오도록 유지.
    """
    if not bottleneck:
        return False
    data = _load("mentees.json")
    for r in data["records"]:
        if r.get("mentee_id") == mentee_id:
            blist = r.setdefault("persistent_bottlenecks", [])
            if bottleneck in blist:
                blist.remove(bottleneck)       # 기존 위치 제거
            blist.insert(0, bottleneck)        # 최신 항목 맨 앞
            r["updated_at"] = now_str()
            _save("mentees.json", data)
            return True
    return False


def update_mentee_transition_profile(mentee_id: str, updates: dict) -> bool:
    """
    transition_profile 서브필드를 부분 갱신.
    session 값이 있는 경우에만 반영 (빈 값 무시).
    """
    if not updates:
        return False
    data = _load("mentees.json")
    for r in data["records"]:
        if r.get("mentee_id") == mentee_id:
            tp = r.setdefault("transition_profile", {})
            for k, v in updates.items():
                if v:  # 빈 값은 덮어쓰지 않음
                    tp[k] = v
            r["updated_at"] = now_str()
            _save("mentees.json", data)
            return True
    return False


def increment_mentee_session_count(mentee_id: str) -> bool:
    data = _load("mentees.json")
    for r in data["records"]:
        if r.get("mentee_id") == mentee_id:
            stats = r.setdefault("session_stats", {"total_sessions": 0, "last_session_at": None})
            stats["total_sessions"] = stats.get("total_sessions", 0) + 1
            stats["last_session_at"] = now_str()
            r["updated_at"] = now_str()
            _save("mentees.json", data)
            return True
    return False


# ─────────────────────────────────────────
# 질문 세션 (Agent 1 세션 출력)
# ─────────────────────────────────────────

def get_question_session(session_id: str) -> dict | None:
    return get_by_id("question_sessions.json", "session_id", session_id)


def create_question_session(
    mentee_id: str,
    refined_question: str,
    conversation_summary: str,
    safe_context: str = "",
    search_query: str | None = None,
    match_query: str | None = None,
    current_bottleneck: str = "",
    expected_answer_type: str = "",
    question_units: list[dict] | None = None,
    taxonomy_tags: dict | None = None,
    routing_hints: dict | None = None,
    hard_case_flags: dict | None = None,
) -> dict:
    """
    Agent 1 종료 후 호출 — 세션 진단 필드를 question_sessions.json에 저장.
    session_id를 반환하므로 Agent 2/3/4 호출 시 이 ID를 사용한다.
    """
    record = {
        "session_id":           new_id("ses_"),
        "mentee_id":            mentee_id,
        "mentor_id":            None,
        "answer_id":            None,

        "refined_question":     refined_question,
        "conversation_summary": conversation_summary,
        "safe_context":         safe_context or conversation_summary,
        "search_query":         search_query or refined_question,
        "match_query":          match_query  or refined_question,

        "current_bottleneck":   current_bottleneck,
        "expected_answer_type": expected_answer_type,
        "question_units":       question_units  or [],
        "taxonomy_tags":        taxonomy_tags   or {},
        "routing_hints":        routing_hints   or {},

        "hard_case_flags":      hard_case_flags or {
            "requires_artifact_review": False,
            "recency_sensitive":        False,
            "scope_too_broad":          False,
            "risk_flags":               [],
            "question_structure":       "",
            "document_help_type":       "",
            "recency_level":            "",
            "recency_reason":           "",
            "source_role":              "",
            "target_role":              "",
            "target_role_specificity":  "unclear",
            "bridge_hypothesis":        "",
            "transferable_skills":      [],
            "target_domain_candidates": [],
        },

        "answer_status":          "pending",
        "llm_direct_answer":      None,
        "llm_partial_answer":     None,
        "llm_feedback_score":     None,
        "retrieval_log":          {},
        "mentor_fallback_reason": None,
        "fallback_type":          None,
        "mentor_match_hints":     {},

        "created_at": now_str(),
        "closed_at":  None,
    }
    return insert("question_sessions.json", record)


def update_session(session_id: str, updates: dict) -> bool:
    """
    question_sessions.json 우선 업데이트.
    레거시 mentee_sessions.json에도 폴백하지 않음 (v2부터 완전 분리).
    """
    return update_by_id("question_sessions.json", "session_id", session_id, updates)


def get_session(session_id: str) -> dict | None:
    """하위 호환 래퍼 — get_question_session() 권장."""
    return get_question_session(session_id)


# 레거시: mentee_sessions.json 접근이 필요한 경우 (마이그레이션 과도기용)
def _legacy_create_session(mentee_id: str, refined_question: str,
                            conversation_summary: str, **kwargs) -> dict:
    """DEPRECATED: create_question_session() 사용 권장."""
    return create_question_session(mentee_id, refined_question, conversation_summary, **kwargs)


# ─────────────────────────────────────────
# 멘토 (장기 프로필)
# ─────────────────────────────────────────

def get_all_mentors() -> list[dict]:
    return get_all("mentors.json")


def get_mentor(mentor_id: str) -> dict | None:
    return get_by_id("mentors.json", "mentor_id", mentor_id)


def save_mentor(record: dict) -> dict:
    return upsert("mentors.json", "mentor_id", record)


def get_mentor_experiences(mentor_id: str) -> list[dict]:
    return [r for r in get_all("mentor_experiences.json") if r["mentor_id"] == mentor_id]


def save_mentor_experience(record: dict) -> dict:
    return upsert("mentor_experiences.json", "experience_id", record)


# ─────────────────────────────────────────
# 멘토 답변 (자산 DB)
# ─────────────────────────────────────────

def get_all_answers() -> list[dict]:
    return get_all("mentor_answers.json")


def get_assetized_answers() -> list[dict]:
    return [r for r in get_all_answers() if r.get("is_assetized")]


def save_answer(record: dict) -> dict:
    return upsert("mentor_answers.json", "answer_id", record)


def create_answer(session_id: str, mentor_id: str, question_content: str,
                  answer_content: str, answer_summarize: str,
                  domain_tags: list[str]) -> dict:
    record = {
        "answer_id": new_id("ans_"),
        "session_id": session_id,
        "mentor_id": mentor_id,
        "question_content": question_content,
        "answer_content": answer_content,
        "answer_summarize": answer_summarize,
        "domain_tags": domain_tags,
        "embedding": None,
        "is_assetized": False,
        "reuse_count": 0,
        "satisfaction_score": None,
        "created_at": now_str(),
    }
    return insert("mentor_answers.json", record)
