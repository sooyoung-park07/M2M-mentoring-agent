"""
실제 잇다 멘토링 50건 Q&A → mentor_answers.json 씨딩 스크립트
generate_scenarios.py(합성 데이터)를 대체

실행:
  python data/seed_real_cases.py               # 기존 누적
  python data/seed_real_cases.py --replace     # 기존 records 삭제 후 새로 씨딩
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.json_db import DB_DIR, new_id, now_str, get_all
from utils.embedding import get_embedding

# ─────────────────────────────────────────────────────────────────
# 1. docx 파싱 — 질답 시나리오.docx에서 50건 추출
# ─────────────────────────────────────────────────────────────────

DOCX_PATH = ROOT / "질답 시나리오.docx"

DOMAIN_MAP = {
    "회계": "회계/재무/금융",
    "재무": "회계/재무/금융",
    "금융": "회계/재무/금융",
    "it": "IT개발/데이터",
    "개발": "IT개발/데이터",
    "데이터": "IT개발/데이터",
    "마케팅": "마케팅/MD",
    "md": "마케팅/MD",
    "전략": "전략기획",
    "기획": "전략기획",
    "연구": "연구/설계",
    "설계": "연구/설계",
    "생산": "생산/품질/제조",
    "품질": "생산/품질/제조",
    "제조": "생산/품질/제조",
    "디자인": "디자인/예술",
    "예술": "디자인/예술",
    "유통": "유통/무역/구매",
    "무역": "유통/무역/구매",
    "구매": "유통/무역/구매",
    "홍보": "홍보/CSR",
    "csr": "홍보/CSR",
    "영업": "영업",
}


def normalize_domain(raw: str) -> list[str]:
    """원본 domain 문자열 → 정제된 태그 리스트"""
    raw_lower = raw.lower()
    tags = set()
    for key, tag in DOMAIN_MAP.items():
        if key in raw_lower:
            tags.add(tag)
    return list(tags) if tags else [raw.strip("() ")]


def extract_first_sentence(text: str) -> str:
    """답변에서 첫 실질 문장 추출 (요약용)"""
    # 인사말 건너뛰고 실질 내용 찾기
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        # 인사·호칭 패턴 제외
        if re.match(r'^(안녕|반갑|감사|멘티님|질문|부족|먼저)', line):
            continue
        if len(line) > 20:
            # 첫 문장만 (마침표 기준)
            sent = line.split("。")[0].split(". ")[0]
            if len(sent) > 15:
                return sent[:80]
    return lines[0][:80] if lines else ""


def parse_docx(path: Path) -> list[dict]:
    """docx 파싱 → 50개 케이스 반환 [{case_id, case_num, domain, title, mentee_question, mentor_answer}]"""
    try:
        import docx
    except ImportError:
        print("python-docx가 없습니다. 설치: pip install python-docx --break-system-packages")
        sys.exit(1)

    doc = docx.Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs)

    # 케이스 헤더 분리
    pattern = re.compile(r'(#\s*case\s*\d+)', re.IGNORECASE)
    parts = pattern.split(full_text)

    cases = []
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""

        num_match = re.search(r'\d+', header)
        case_num = int(num_match.group()) if num_match else len(cases) + 1

        # 분야
        domain_match = re.search(r'분야[:\s]*(.+)', body)
        domain_raw = domain_match.group(1).strip() if domain_match else ""
        # 첫 줄에 있는 경우도 처리
        if not domain_raw:
            first_lines = body.strip().split('\n')[:3]
            for line in first_lines:
                if '분야' in line:
                    domain_raw = re.sub(r'분야[:\s]*', '', line).strip()
                    break

        # 주제
        title_match = re.search(r'주제[:\s]*(.+)', body)
        title = title_match.group(1).strip() if title_match else f"CASE_{case_num:03d}"

        # 멘티 질문
        q_match = re.search(
            r'(?:멘티\s*질문|멘티질문)[:\s]*([\s\S]+?)(?=멘토\s*답변|토\s*답변|$)',
            body
        )
        mentee_question = q_match.group(1).strip() if q_match else ""

        # 멘토 답변
        a_match = re.search(
            r'(?:멘토\s*답변|토\s*답변)[:\s]*([\s\S]+?)(?=#\s*case\s*\d+|$)',
            body,
            re.IGNORECASE
        )
        mentor_answer = a_match.group(1).strip() if a_match else ""

        # 질문 fallback: 주제 다음~멘토 답변 전
        if not mentee_question and title:
            fb_match = re.search(
                rf'{re.escape(title)}([\s\S]+?)(?:멘토\s*답변|토\s*답변)',
                body
            )
            if fb_match:
                mentee_question = fb_match.group(1).strip()

        if mentee_question or mentor_answer:
            cases.append({
                "case_id": f"CASE_{case_num:03d}",
                "case_num": case_num,
                "domain": domain_raw,
                "title": title,
                "mentee_question": mentee_question,
                "mentor_answer": mentor_answer,
            })

        i += 2

    return cases


# ─────────────────────────────────────────────────────────────────
# 2. DB 헬퍼
# ─────────────────────────────────────────────────────────────────

def _load_db(filename: str) -> dict:
    path = DB_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _save_db(filename: str, data: dict):
    path = DB_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_mentor_ids() -> list[str]:
    try:
        mentors = get_all("mentors.json")
        return [m["mentor_id"] for m in mentors]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────
# 3. 씨딩 메인
# ─────────────────────────────────────────────────────────────────

def build_record(case: dict, mentor_id: str) -> dict | None:
    """케이스 1건 → mentor_answers 레코드"""
    mentee_q = case["mentee_question"]
    mentor_a = case["mentor_answer"]

    if not mentee_q or not mentor_a:
        return None

    domain_tags = normalize_domain(case["domain"])
    title = case["title"]

    # answer_summarize: 답변 첫 실질 문장
    answer_summarize = extract_first_sentence(mentor_a)

    # 임베딩: 제목(proxy refined_question) + 답변 요약
    embed_text = f"{title}\n{answer_summarize}"
    try:
        embedding = get_embedding(embed_text)
    except Exception as e:
        print(f"  임베딩 오류: {e}")
        return None

    return {
        "answer_id":       new_id("ans_"),
        "session_id":      f"real_{case['case_id'].lower()}",
        "mentor_id":       mentor_id,
        "question_content": title,           # 제목 = proxy refined question
        "answer_content":  mentor_a,
        "answer_summarize": answer_summarize,
        "domain_tags":     domain_tags,
        "embedding":       embedding,
        "is_assetized":    True,
        "reuse_count":     0,
        "satisfaction_score": None,
        "created_at":      now_str(),
        # 원본 보존
        "_case_id":        case["case_id"],
        "_case_title":     title,
        "_mentee_question": mentee_q,
    }


def seed(replace: bool = False):
    print(f"질답 시나리오.docx 파싱 중: {DOCX_PATH}")
    if not DOCX_PATH.exists():
        print(f"  파일 없음: {DOCX_PATH}")
        sys.exit(1)

    cases = parse_docx(DOCX_PATH)
    print(f"  파싱 완료: {len(cases)}건\n")

    mentor_ids = get_mentor_ids()
    if not mentor_ids:
        print("  경고: mentors.json에 mentor가 없음. 더미 ID 사용.")
        mentor_ids = [new_id("mr_")]

    records = []
    for idx, case in enumerate(cases, 1):
        mentor_id = mentor_ids[(idx - 1) % len(mentor_ids)]
        print(f"[{idx:02d}/{len(cases)}] {case['case_id']} | {case['title'][:30]}...")

        record = build_record(case, mentor_id)
        if record:
            records.append(record)
            print(f"  ✓ 임베딩 생성 완료 (domain: {record['domain_tags']})")
        else:
            print(f"  ✗ 건너뜀 (질문 또는 답변 없음)")

    # DB 저장
    db = _load_db("mentor_answers.json")
    if replace:
        db["records"] = records
        print(f"\n✓ mentor_answers.json 교체 완료: {len(records)}건")
    else:
        db["records"].extend(records)
        print(f"\n✓ mentor_answers.json 추가 완료: {len(records)}건 (총 {len(db['records'])}건)")
    _save_db("mentor_answers.json", db)

    # mentee_sessions 클리어
    sessions_db = _load_db("mentee_sessions.json")
    sessions_db["records"] = []
    _save_db("mentee_sessions.json", sessions_db)
    print("✓ mentee_sessions.json 클리어 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true",
                        help="기존 records 삭제 후 실제 50건으로 교체 (기본: 누적 추가)")
    args = parser.parse_args()
    seed(replace=args.replace)
