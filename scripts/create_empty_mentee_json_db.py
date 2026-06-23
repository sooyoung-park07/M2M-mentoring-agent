import json
from pathlib import Path


# ============================================================
# 0. 경로 설정
# ============================================================

# 현재 파일 위치:
# 프로젝트2팀/data_db/scripts/create_empty_mentee_json_db.py
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DB_DIR = SCRIPT_DIR.parent

# JSON DB 저장 위치:
# 프로젝트2팀/data_db/json_db/
JSON_DB_DIR = DATA_DB_DIR / "json_db"
JSON_DB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 빈 JSON DB 스키마 정의
# ============================================================

mentees_db = {
    "schema_name": "mentees",
    "description": "멘티 기본 정보 DB",
    "fields": {
        "mentee_id": "string",
        "mentee_info": {
            "name": "string",
            "gender": "string",
            "age": "int"
        },
        "background": {
            "school": "string",
            "major": "string",
            "grade": "string or null",
            "status": "string"
        },
        "considering_options": {
            "graduate_school": "bool",
            "internship": "bool",
            "full_time": "bool"
        },
        "target_goal_extracted": {
            "primary": "string",
            "secondary": "list[string]"
        },
        "interest_domain": "list[string]",
        "total_session": "int",
        "matching_summary_text": "string",
        "be_go": "string",
        "active": "bool"
    },
    "records": []
}


mentee_experiences_db = {
    "schema_name": "mentee_experiences",
    "description": "멘티 경험 정보 DB",
    "fields": {
        "experience_id": "string",
        "mentee_id": "string",
        "experience_type": "project | internship | work | club | course | certification | award | education | etc",
        "title": "string",
        "description": "string",
        "organization": "string",
        "start_date": "YYYY-MM",
        "end_date": "YYYY-MM or present",
        "role": "string",
        "key_skills": "string",
        "tools": "string"
    },
    "records": []
}


mentee_sessions_db = {
    "schema_name": "mentee_sessions",
    "description": "멘티 세션/질문 정보 DB",
    "fields": {
        "session_id": "string",
        "mentee_id": "string",
        "mentor_id": "string or null",
        "refined_question": "string",
        "conversation_summary": "string",
        "matching_status": "bool",
        "mentor_answer": "string",
        "created_at": "YYYY-MM-DD HH:MM",
        "closed_session": "bool",
        "closed_at": "YYYY-MM-DD HH:MM or null"
    },
    "records": []
}


# ============================================================
# 2. 저장 함수
# ============================================================

def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 3. 메인 실행
# ============================================================

def main():
    save_json(JSON_DB_DIR / "mentees.json", mentees_db)
    save_json(JSON_DB_DIR / "mentee_experiences.json", mentee_experiences_db)
    save_json(JSON_DB_DIR / "mentee_sessions.json", mentee_sessions_db)

    print("빈 JSON 멘티 DB 생성 완료")
    print(f"저장 위치: {JSON_DB_DIR}")
    print("- mentees.json")
    print("- mentee_experiences.json")
    print("- mentee_sessions.json")


if __name__ == "__main__":
    main()
