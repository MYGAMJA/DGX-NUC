# 프로젝트 가이드 (Claude / 기타 AI 도구용)

## 읽지 말아야 할 경로

### `BHL/CALIBRATION/electronic_calib/backup/`

모터 전기적 캘리브레이션(electrical flux offset cal)의 **과거 결과 스냅샷**들이 모여 있는 폴더.

- **사용자가 명시적으로 요청하지 않는 한 읽지 말 것.**
- before/after 페어들의 실제 차이는 `encoder.flux_offset` 한 줄뿐이라 정보 밀도가 매우 낮음.
- 과거 시점의 값들이라 현재 BESC flash에 들어 있는 값과 일치한다는 보장 없음.
- 이 폴더를 무비판적으로 참조하면 "지금 상태"를 잘못 추정하는 사례가 반복됨.

**현재 모터 상태가 궁금하면**:
- 코드 진실: `BHL/CALIBRATION/electronic_calib/calibrate_one.py` 가 캘리브 후 새 JSON을 `BHL/CALIBRATION/electronic_calib/` 직속에 씀
- 디바이스 진실: CAN 버스로 BESC에 직접 ping/read (`BHL/CALIBRATION/can_sweep/`, `BHL/scripts/` 참고)
- 작업 기록: `BHL/docs/` (날짜별 MD)

사용자가 "백업 파일도 봐줘" 같은 명시적 요청을 했을 때에만 `backup/` 를 읽을 것.

## BHL 공식 문서 로컬 미러

Berkeley Humanoid Lite GitBook(https://berkeley-humanoid-lite.gitbook.io/docs) 의 27개 페이지 전체가 `BHL/reference/BHL_official_docs/` 에 보존되어 있다.

- **언제 가야 하나**: 사용자가 "공식 문서에", "BHL 공식", "GitBook 에" 같은 표현을 쓸 때, 또는 FOC/CAN 프로토콜/joint ID 매핑/모터 파라미터/펌웨어 플래시 절차 등 공식 스펙이 필요한 질문이 나올 때
- **먼저 읽을 파일**: [`BHL/reference/BHL_official_docs/README.md`](BHL/reference/BHL_official_docs/README.md) — 어떤 질문이 어느 파일로 가야 하는지 라우팅 가이드
- **공식 vs 본 프로젝트 현실 차이**: [`BHL/reference/BHL_official_docs/_CROSS_REFERENCE.md`](BHL/reference/BHL_official_docs/_CROSS_REFERENCE.md) — 공식 문서 값과 본 프로젝트 실제 상태가 다른 항목들 (캘리 전류, phase_order flash 버그, 5010 370KV 누락 등). 공식 문서 인용 전 반드시 확인.

폴더 안 파일들은 `00_*` 부터 `90_*` 까지 번호 순으로 카테고리별 정렬되어 있고, 각 파일 상단 HTML 주석에 원본 URL과 sync 날짜가 박혀있다.
