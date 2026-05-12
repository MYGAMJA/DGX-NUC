# 모터 컨트롤러 펌웨어 플래싱 - Step by Step 가이드 (macOS)

> 🎯 **목표**: Berkeley Humanoid Lite의 모터 컨트롤러 보드(B-G431B-ESC1)에 펌웨어를 플래싱한다
> ⏱️ **예상 소요 시간**: 30분 ~ 1시간 (처음 하는 경우)
> 💻 **필요한 것**: macOS (Intel 또는 Apple Silicon), 모터 컨트롤러 보드, Micro USB **데이터** 케이블, (필요 시) USB-C ↔ USB-A 어댑터 또는 USB-C ↔ Micro USB 케이블

---

## 📌 시작하기 전에 꼭 읽어보세요

이 가이드는 **한 번도 해본 적 없는 사람도 따라할 수 있도록** 만든 거예요. 그래서 길지만, 빠뜨리는 단계가 없도록 자세히 적었어요. 빠르게 끝내고 싶다고 **단계 건너뛰지 마세요!** 특히 "⚠️ 주의" 표시는 꼭 읽어주세요.

### macOS 환경 사전 점검

- **macOS 버전**: 10.15(Catalina) 이상 권장. 최신 STM32CubeIDE는 macOS 12 이상 권장
- **Apple Silicon(M1/M2/M3) 사용자**: 최신 버전(1.13 이상)은 ARM64 네이티브 지원. 구버전을 쓰면 Rosetta 2 설치 필요
- **포트**: 요즘 맥북은 USB-C만 있으므로 **USB-C ↔ Micro USB 케이블** 또는 **USB-A → USB-C 허브/어댑터** 준비
- **케이블**: 반드시 **데이터 전송 가능한 케이블**. 충전 전용 케이블은 보드 인식 안 됨!

### 전체 흐름 미리보기

```
1. STM32CubeIDE 설치 (macOS용)
       ↓
2. 펌웨어 코드 다운로드
       ↓
3. 프로젝트 열기
       ↓
4. 빌드 에러 수정
       ↓
5. 보드 연결
       ↓
6. Run #1 (초기 설정)
       ↓
7. Run #2 (CAN ID 및 모터 설정)
       ↓
8. Run #3 (최종 확정)
       ↓
   완료! 🎉
```

플래싱은 **총 3번** "Run" 버튼을 눌러야 끝나요. 각 단계마다 설정값을 바꿔야 하니까 헷갈리지 않게 주의하세요!

---

## 1단계: STM32CubeIDE 설치 (macOS)

### 1-1. 다운로드

1. [STM32CubeIDE 공식 사이트](https://www.st.com/en/development-tools/stm32cubeide.html) 접속
2. **macOS 버전** (`STM32CubeIDE-Mac` / `.dmg`) 다운로드
   - Apple Silicon이면 `arm64` 또는 `Universal` 버전 권장
   - Intel 맥이면 `x86_64` 버전
3. ST 계정 만들기 (이메일 인증 필요)
4. 인증 메일에서 다운로드 링크 받아서 다운로드

> ⏱️ 다운로드 파일이 큼 (~1GB). 안정적인 인터넷에서 받으세요.

### 1-2. 설치 시 발생할 수 있는 에러 (macOS)

#### "확인되지 않은 개발자" 경고가 뜰 때

`.dmg` 파일을 열거나 앱을 처음 실행할 때 **"확인되지 않은 개발자가 만든 앱이므로 열 수 없습니다"** 가 뜰 수 있어요.

- **방법 1 (권장)**: Finder에서 앱 아이콘을 **Control + 클릭** (또는 우클릭) → **"열기"** 선택 → 다이얼로그에서 다시 **"열기"** 클릭
- **방법 2**: **시스템 설정 → 개인정보 보호 및 보안** → 아래쪽 차단 메시지 옆 **"확인 없이 열기"** 클릭
- **방법 3 (터미널)**: 격리 속성 제거
  ```bash
  xattr -dr com.apple.quarantine /Applications/STM32CubeIDE.app
  ```

#### "손상되어 열 수 없습니다" 에러

```bash
sudo xattr -dr com.apple.quarantine ~/Downloads/st-stm32cubeide_*.dmg
```

위 명령으로 격리 속성을 제거한 뒤 다시 열어보세요.

### 1-3. 설치

1. 다운로드한 `.dmg` 파일을 더블클릭
2. 안내 창에서 **STM32CubeIDE.app** 아이콘을 **Applications 폴더로 드래그**
3. 처음 실행은 위 1-2의 "Control + 클릭 → 열기" 방식으로

✅ **완료 체크**: STM32CubeIDE가 실행되고 "Welcome to STM32CubeIDE" 화면이 보이면 성공

---

## 2단계: 펌웨어 코드 다운로드

### 2-1. GitHub에서 ZIP 다운로드

1. [Recoil-Motor-Controller-BESC 레포지토리](https://github.com/T-K-233/Recoil-Motor-Controller-BESC) 접속
2. 녹색 **"Code"** 버튼 클릭
3. **"Download ZIP"** 클릭

> 💡 터미널이 익숙하면 git clone도 OK:
> ```bash
> cd ~/Documents
> git clone https://github.com/T-K-233/Recoil-Motor-Controller-BESC.git
> ```

### 2-2. 압축 해제

⚠️ **중요**: 압축 푸는 위치가 중요해요!

- ✅ **좋은 경로**: `~/Documents/Recoil/`, `~/Projects/Recoil/`, `/Users/사용자명/Recoil/`
- ❌ **나쁜 경로**: 한글이 포함된 경로(예: `~/Documents/문서/...`), 공백이 많은 긴 경로, iCloud Drive 안의 폴더(동기화 충돌 가능), 외장 디스크의 NTFS 파티션

> 💡 macOS는 기본적으로 ZIP을 더블클릭하면 같은 위치에 풀어요. 한글/공백 없는 경로로 먼저 옮긴 뒤 풀면 안전합니다.

### 2-3. 폴더 구조 확인

압축 풀고 나면 이런 구조가 보여야 해요:

```
Recoil-Motor-Controller-BESC-main/
└── Recoil-Motor-Controller-B-G431B-ESC1/  ← 이 폴더가 우리가 쓸 프로젝트
    ├── Core/
    ├── Drivers/
    └── ...
```

터미널 확인 명령:
```bash
ls ~/Documents/Recoil/Recoil-Motor-Controller-BESC-main/
```

✅ **완료 체크**: 위 구조가 보이면 성공

---

## 3단계: STM32CubeIDE에서 프로젝트 열기

### 3-1. STM32CubeIDE 실행

Launchpad 또는 `Applications` 폴더에서 **STM32CubeIDE** 실행.

처음 실행하면 Workspace 위치를 묻는데, **기본값 그대로 두고 Launch** 클릭.

> 💡 Workspace 기본 경로는 보통 `/Users/사용자명/STM32CubeIDE/workspace_X.X` 입니다.

### 3-2. Information Center 탭 닫기

환영 화면이 뜨면 상단의 **"Information Center" 탭의 X**를 눌러 닫으세요.

### 3-3. 프로젝트 import

1. 상단 메뉴 **File → Open Projects from File System...**
2. **Directory...** 버튼 클릭
3. Finder 창이 뜨면 압축 푼 폴더 안의 **`Recoil-Motor-Controller-B-G431B-ESC1`** 폴더 선택
4. **Finish** 클릭

⚠️ **주의**: 왼쪽의 "Create / Import STM32 project" 버튼은 새 프로젝트 만들 때 쓰는 거예요. **누르지 마세요!**

✅ **완료 체크**: 왼쪽 Project Explorer에 프로젝트 폴더 트리가 보이면 성공

---

## 4단계: 빌드 에러 수정 ⚠️ 필수!

이 단계를 건너뛰면 빌드가 실패해요!

### 4-1. 빌드 시도

상단 **망치 🔨 아이콘**을 클릭해서 빌드 시도. 그러면 이런 에러가 뜰 거예요:

```
../Core/Src/motor_controller.c:98:9: error: implicit declaration of function 'sprintf'
```

**당황하지 마세요!** 이건 이미 알려진 문제이고 쉽게 고칠 수 있어요.

### 4-2. 파일 열기

왼쪽 Project Explorer에서:

```
Core → Src → motor_controller.c
```

이 파일을 더블클릭해서 여세요.

### 4-3. 한 줄 추가하기

파일 맨 위쪽에 가면 이런 줄이 있을 거예요:

```c
#include "motor_controller.h"
```

이 위에 한 줄 추가:

```c
#include <stdio.h>
#include "motor_controller.h"
```

### 4-4. 저장 + 다시 빌드

- **⌘ Command + S** 로 저장
- 다시 **망치 🔨 아이콘** 클릭

✅ **완료 체크**: 하단 Console에 `Build Finished. 0 errors` 메시지가 보이면 성공
(warnings는 무시해도 OK)

---

## 5단계: 설정 파일 위치 확인

이제 핵심 설정 파일을 알고 있어야 해요. 앞으로 자주 열게 될 거예요.

### 5-1. 파일 열기

왼쪽 Project Explorer에서:

```
Core → Inc → motor_controller_conf.h
```

이 파일을 더블클릭해서 여세요.

### 5-2. 자주 볼 라인들

**⌘ Command + F** 로 검색하면서 위치를 미리 알아두세요:

| 검색어 | 무엇인지 |
|---|---|
| `DEVICE_CAN_ID` | 이 보드의 CAN ID (관절마다 다름) |
| `FIRST_TIME_BOOTUP` | 처음 부팅 플래그 |
| `LOAD_ID_FROM_FLASH` | Flash에서 ID 읽기 플래그 |
| `LOAD_CONFIG_FROM_FLASH` | Flash에서 설정 읽기 플래그 |
| `MOTORPROFILE_` | 모터 종류 선택 |

✅ **완료 체크**: 위 5개 위치를 다 찾을 수 있으면 OK

---

## 6단계: CAN ID 정하기

### 6-1. 어떤 관절에 쓸 보드인지 정하기

각 보드는 **하나의 관절**용이에요. 이 보드를 어느 관절에 쓸지 정해야 해요.

### 6-2. CAN ID 표 (왼쪽 다리)

| 관절 이름 | CAN ID |
|---|---|
| left_hip_roll | **1** |
| left_hip_yaw | **3** |
| left_hip_pitch | **5** |
| left_knee_pitch | **7** |
| left_ankle_pitch | **11** |
| left_ankle_roll | **13** |

### 6-3. CAN ID 표 (오른쪽 다리)

| 관절 이름 | CAN ID |
|---|---|
| right_hip_roll | **2** |
| right_hip_yaw | **4** |
| right_hip_pitch | **6** |
| right_knee_pitch | **8** |
| right_ankle_pitch | **12** |
| right_ankle_roll | **14** |

> 💡 **규칙**: 왼쪽은 **홀수**, 오른쪽은 **짝수**

### 6-4. 사용할 모터 확인

모터에 따라 다른 프로파일을 선택해야 해요:

| 관절 종류 | 사용 모터 | 선택할 프로파일 |
|---|---|---|
| 큰 관절 (hip, knee 등) | M6C12 150KV | `MOTORPROFILE_MAD_M6C12_150KV` |
| 작은 관절 (ankle 등) | MAD 5010 110KV | `MOTORPROFILE_MAD_5010_110KV` |

**보드마다 다를 수 있으니, 조립 BOM 또는 모터 라벨을 확인하세요!**

---

## 7단계: Run #1 - 초기 Flash Option Byte 설정

### 7-1. 설정 파일 수정

`motor_controller_conf.h`에서 다음을 확인/수정:

```c
#define FIRST_TIME_BOOTUP               1   // ⭐ 1로 설정!
```

다른 플래그들은 그대로 두세요.

### 7-2. 보드 연결

- 모터 컨트롤러 보드에 **Micro USB 데이터 케이블 연결**
- 맥북 USB-C 포트에 꽂기 (어댑터/허브 필요할 수 있음)
- 보드의 빨간 LED가 켜지면 전원 정상

> 💡 USB 인식 확인 (터미널):
> ```bash
> ls /dev/cu.usbmodem*
> ```
> ST-LINK가 보이면 인식된 거예요. macOS는 별도 드라이버 없이 잡힙니다.

### 7-3. 저장

**⌘ Command + S** 로 파일 저장

### 7-4. 프로젝트 선택 후 Run ⚠️

⚠️ **이 부분이 헷갈리니까 잘 따라하세요!**

1. 왼쪽 Project Explorer에서 **프로젝트 루트** (`Recoil-Motor-Controller-B-G431B-ESC1`) 를 **한 번 클릭**해서 선택
   - 코드 파일이 아니라 **프로젝트 폴더 자체**를 선택!
2. 상단 메뉴 **Run → Run As → STM32 C/C++ Application** 클릭

> ❓ **왜 이렇게 해야 하나?** 그냥 Run 버튼 누르면 "Unable To Launch" 에러가 뜰 수 있어요. 위 방법대로 하면 IDE가 자동으로 실행 설정을 만들어줘요.

### 7-5. ST-LINK 펌웨어 업데이트 (새 보드일 때)

처음 쓰는 보드면 **"STLinkUpgrade" 창**이 뜰 거예요.

1. **"Open in update mode"** 클릭
2. **"Upgrade"** 버튼 클릭
3. ⚠️ 업데이트 진행 중 **USB 뽑지 마세요!** 10~30초 소요
4. 완료되면 창 닫기 (빨간 X)
5. 다시 **Run → Run As → STM32 C/C++ Application** 클릭

> 💡 macOS에서 ST-LINK 펌웨어 업데이트가 안 열린다면, 보안 설정에서 권한을 허용한 뒤 다시 시도하세요.
> **시스템 설정 → 개인정보 보호 및 보안** 에서 차단된 항목 확인.

### 7-6. Launch Configuration Selection 창

두 가지 옵션이 보이면:

- ✅ **Recoil-Motor-Controller-B-G431B-ESC1** ← 이거 선택
- ❌ **Recoil-Motor-Controller-B-G431B-ESC1 Debug** ← 이건 선택 X

**OK** 클릭

### 7-7. 결과 확인

하단 Console에:

```
Download verified successfully
```

이 메시지가 보이면 Run #1 성공! ✅

---

## 8단계: Run #2 - CAN ID 및 모터 설정

### 8-1. USB 케이블 뽑았다 다시 꽂기 🔌

⚠️ **이 단계 꼭 해야 함!**

- 보드에서 USB 케이블 **뽑기**
- 2~3초 기다리기
- **다시 꽂기**

이렇게 해야 Run #1에서 설정한 Flash option이 보드에 적용돼요.

### 8-2. `motor_controller_conf.h` 수정

**⌘ Command + F** 로 검색하면서 다음 값들을 수정:

#### 변경 1: FIRST_TIME_BOOTUP을 0으로

```c
#define FIRST_TIME_BOOTUP               0   // 1 → 0
```

#### 변경 2: CAN ID 설정

```c
#define DEVICE_CAN_ID                   1   // ⭐ 6단계에서 정한 ID로!
```

> 💡 예: left_hip_roll이면 1, left_hip_yaw면 3, ...

#### 변경 3: LOAD 플래그들을 0으로

```c
#define LOAD_ID_FROM_FLASH              0   // 1 → 0
#define LOAD_CONFIG_FROM_FLASH          0   // 1 → 0
```

> ⚠️ **LOAD_CALIBRATION_FROM_FLASH**는 1 그대로 두세요!

#### 변경 4: 모터 프로파일 활성화

파일 아래쪽으로 스크롤하면 이런 부분이 있어요:

```c
/** ======== Motor Selection ======== **/

// uncomment the motor that you are using
//#define MOTORPROFILE_MAD_M6C12_150KV
//#define MOTORPROFILE_MAD_5010_110KV
//#define MOTORPROFILE_MAD_5010_310KV
//#define MOTORPROFILE_MAD_5010_370KV
```

**해당하는 모터의 `//`를 제거**하세요. 예시 (M6C12 사용 시):

```c
#define MOTORPROFILE_MAD_M6C12_150KV   // ← //가 제거됨
//#define MOTORPROFILE_MAD_5010_110KV
//#define MOTORPROFILE_MAD_5010_310KV
//#define MOTORPROFILE_MAD_5010_370KV
```

> ⚠️ **한 개만** uncomment하세요. 여러 개 풀면 안 됩니다!

### 8-3. 저장 + Run

- **⌘ Command + S** 저장
- **녹색 ▶ Run 버튼** 클릭 (또는 Run → Run)

### 8-4. 결과 확인

Console에 `Download verified successfully` 보이면 Run #2 성공! ✅

---

## 9단계: Run #3 - 최종 확정

### 9-1. USB 케이블 뽑았다 다시 꽂기 🔌

또 한 번! 안 빠뜨리고 해주세요.

### 9-2. `motor_controller_conf.h` 수정

이번엔 다시 LOAD 플래그들을 1로 되돌립니다:

```c
#define FIRST_TIME_BOOTUP               0   // 그대로 (절대 1로 바꾸지 마세요!)
#define LOAD_ID_FROM_FLASH              1   // 0 → 1
#define LOAD_CONFIG_FROM_FLASH          1   // 0 → 1
#define LOAD_CALIBRATION_FROM_FLASH     1   // 그대로
```

> 💡 **왜 이렇게 하냐?** Run #2에서 Flash에 설정값을 썼어요. 이제 부팅할 때마다 그 Flash 값을 읽어서 쓰도록 만드는 거예요.

### 9-3. 저장 + Run

- **⌘ Command + S** 저장
- **녹색 ▶ Run 버튼** 클릭

### 9-4. 결과 확인

Console에 `Download verified successfully` 보이면 Run #3 성공! 🎉

---

## 10단계: 완료 확인

### 10-1. USB 뽑았다 다시 꽂기

마지막으로 한 번 더!

### 10-2. LED 확인

보드의 LED가 **약 1초에 한 번씩 깜빡이면** 정상 동작 중!

> ⚠️ **만약 LED가 1Hz로 안 깜빡여도 너무 걱정 마세요.**
>
> - 빨간 PWR LED는 항상 켜져 있는 게 정상이에요
> - 깜빡이는 LED는 보드 안쪽의 작은 SMD LED일 수 있어서 밝은 곳에서는 잘 안 보일 수 있어요
> - 펌웨어 플래싱 자체가 성공했다면 (Console에 `Download verified successfully` 나왔다면) OK
> - 진짜 동작 확인은 모터 연결하고 캘리브레이션할 때 함

---

## 🎉 완료!

이 보드의 펌웨어 플래싱은 끝났어요!

다른 보드도 같은 방식으로 하되, **CAN ID와 모터 프로파일만** 다르게 설정하면 돼요.

---

## 📋 다음 보드 작업용 체크리스트

다른 보드 작업할 때 이 체크리스트만 따라가세요:

### 사전 준비
- [ ] 어느 관절용인지 정함 (CAN ID 결정)
- [ ] 사용할 모터 확인 (M6C12 / 5010 시리즈)
- [ ] USB-C 어댑터/케이블 준비

### Run #1
- [ ] `motor_controller_conf.h` 열기
- [ ] `FIRST_TIME_BOOTUP = 1` 설정
- [ ] 저장 (⌘ + S)
- [ ] 보드를 USB로 연결
- [ ] **Run → Run As → STM32 C/C++ Application**
- [ ] Console에 `Download verified successfully` 확인

### Run #2
- [ ] USB 뽑고 다시 꽂기 🔌
- [ ] `FIRST_TIME_BOOTUP = 0`
- [ ] `DEVICE_CAN_ID = [관절 ID]`
- [ ] `LOAD_ID_FROM_FLASH = 0`
- [ ] `LOAD_CONFIG_FROM_FLASH = 0`
- [ ] 해당 모터 프로파일 `//` 제거
- [ ] 저장
- [ ] **녹색 ▶ Run 버튼** 클릭
- [ ] Console에 `Download verified successfully` 확인

### Run #3
- [ ] USB 뽑고 다시 꽂기 🔌
- [ ] `LOAD_ID_FROM_FLASH = 1`
- [ ] `LOAD_CONFIG_FROM_FLASH = 1`
- [ ] (FIRST_TIME_BOOTUP은 0 그대로!)
- [ ] 저장
- [ ] **녹색 ▶ Run 버튼** 클릭
- [ ] Console에 `Download verified successfully` 확인

### 완료
- [ ] USB 뽑고 다시 꽂기
- [ ] 보드에 라벨 붙이기 (어느 관절용인지) 💡 추천!

---

## 🆘 트러블슈팅 (macOS)

작업 중 문제가 생기면 여기 확인하세요!

### "확인되지 않은 개발자" / "손상되어 열 수 없습니다" 경고
→ Control + 클릭 → 열기, 또는 터미널에서:
```bash
sudo xattr -dr com.apple.quarantine /Applications/STM32CubeIDE.app
```

### 빌드 에러: `sprintf` implicit declaration
→ 4단계로 가서 `motor_controller.c`에 `#include <stdio.h>` 추가

### "Unable To Launch" 팝업
→ 코드 파일이 아니라 **프로젝트 루트**를 선택한 상태에서 **Run → Run As → STM32 C/C++ Application**

### 보드가 인식 안 됨
→ USB 케이블 확인 (충전 전용 케이블 X, **데이터 케이블** ○)
→ 다른 USB-C 포트 또는 다른 허브/어댑터 시도
→ 터미널에서 `ls /dev/cu.usbmodem*` 로 확인. 비어 있으면 보드/케이블 문제
→ **시스템 정보** (메뉴바 사과 → 이 Mac에 관하여 → 추가 정보 → 시스템 리포트 → USB) 에서 ST-Link가 잡히는지 확인

### ST-LINK 펌웨어 업데이트 창이 안 뜸
→ 이미 업데이트되어 있는 보드라서 그래요. 정상이에요!

### ST-LINK Upgrade 도구가 안 열리거나 권한 차단
→ **시스템 설정 → 개인정보 보호 및 보안** 에서 차단 메시지 확인 후 "확인 없이 열기" / 권한 허용

### Download 실패 / "No ST-Link detected"
→ USB 뽑고 다시 꽂은 후 재시도
→ USB 허브 거치면 불안정할 수 있음. 가능하면 맥북 본체 포트에 직결

### Apple Silicon에서 IDE 실행이 느리거나 죽음
→ 최신 버전(1.13+) 사용 권장
→ 구버전이면 터미널에서 Rosetta 설치 후 재시도:
```bash
softwareupdate --install-rosetta
```

### LED가 1Hz로 안 깜빡임
→ 펌웨어 플래싱이 `Download verified successfully` 떴다면 OK
→ 실제 동작 테스트는 모터 연결 후 캘리브레이션 단계에서 함

---

## 🔗 참고 자료

- [Berkeley Humanoid Lite 공식 문서](https://berkeley-humanoid-lite.gitbook.io/docs)
- [펌웨어 플래싱 가이드 (영어 원문)](https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/flashing-the-motor-controllers)
- [Joint ID Mapping](https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/joint-id-mapping)
- 영상 가이드: [YouTube - hUxj4s9o3TY](https://youtu.be/hUxj4s9o3TY)

---

## 💬 막힐 때 체크할 것들

- ✅ 단계 건너뛰지 않았나? (특히 4단계 sprintf 수정, USB 뽑았다 꽂기)
- ✅ 파일 저장했나? (⌘ Command + S)
- ✅ 플래그 값들 헷갈리지 않았나? (Run마다 다름!)
- ✅ 프로젝트 루트 선택하고 Run 눌렀나? (파일 선택 X)
- ✅ 모터 프로파일 한 개만 uncomment 했나?
- ✅ USB 케이블이 데이터 전송용 맞나? (충전 전용 X)
- ✅ macOS 보안 경고 해제했나?

---

**작성**: 2026-05-12
**대상**: B-G431B-ESC1 모터 컨트롤러 보드
**호스트 OS**: macOS (Intel / Apple Silicon)
**프로젝트**: Berkeley Humanoid Lite

🙌 조원들 모두 화이팅!
