# Git 명령어 참고

## 기본 상태 확인

```bash
# 현재 브랜치 및 변경사항 확인
git status

# 커밋 히스토리
git log --oneline -10

# 원격 브랜치 목록
git branch -r

# 로컬 + 원격 브랜치 모두
git branch -a
```

---

## Pull (가져오기)

```bash
# 원격에서 최신 정보 가져오기 (merge 없음)
git fetch origin

# 현재 브랜치에 원격 최신 커밋 반영
git pull

# 특정 브랜치 pull
git pull origin main

# main의 변경사항을 현재 브랜치에 merge
git fetch origin
git merge origin/main
```

---

## Branch (브랜치)

```bash
# 브랜치 생성
git checkout -b feat/내-브랜치-이름

# 기존 브랜치로 이동
git checkout main
git checkout feat/ethan-merge-no-md

# 원격 브랜치를 로컬로 가져와서 이동
git checkout -b feat/jimmy-sim2sim origin/feat/jimmy-sim2sim
```

---

## Commit & Push

```bash
# 변경 파일 스테이징 (전체)
git add .

# 특정 파일만
git add sim/mujoco/run_sim2sim.sh

# 커밋
git commit -m "feat: 커밋 메시지"

# 원격에 push (처음)
git push -u origin feat/내-브랜치-이름

# 이후 push
git push
```

---

## Pull Request (PR)

GitHub CLI (`gh`) 사용:

```bash
# GitHub CLI 설치 확인
gh --version

# 현재 브랜치 → main으로 PR 생성
gh pr create --base main --title "feat: 제목" --body "설명"

# 드래프트 PR
gh pr create --base main --title "feat: 제목" --draft

# PR 목록 확인
gh pr list

# PR 상태 확인
gh pr status

# PR 상세 보기
gh pr view 7
```

---

## Merge

```bash
# 로컬에서 merge (main 기준)
git checkout main
git merge feat/내-브랜치-이름

# PR을 GitHub CLI로 merge
gh pr merge 7 --merge          # 일반 merge commit
gh pr merge 7 --squash         # squash merge (커밋 하나로 합침)
gh pr merge 7 --rebase         # rebase merge

# merge 후 브랜치 삭제
git branch -d feat/내-브랜치-이름          # 로컬 삭제
git push origin --delete feat/내-브랜치-이름  # 원격 삭제
```

---

## 자주 쓰는 전체 흐름

### 새 기능 작업 시작
```bash
git checkout main
git pull
git checkout -b feat/새기능
# ... 작업 ...
git add .
git commit -m "feat: 새기능 추가"
git push -u origin feat/새기능
gh pr create --base main --title "feat: 새기능 추가"
```

### main의 최신 내용을 내 브랜치에 반영
```bash
git fetch origin
git merge origin/main
# 충돌 해결 후
git add .
git commit -m "chore: merge main into feat/내-브랜치"
git push
```

### 충돌(conflict) 해결
```bash
# 충돌 파일 확인
git status

# 파일 편집 후 해결 완료 표시
git add 충돌파일.py

# merge 완료
git commit
```
