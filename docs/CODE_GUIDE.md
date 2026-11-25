# 코드 설명서: 파일별 안내

이 저장소의 모든 코드가 **무엇을 하는지, 어떻게 실행하는지, 무엇과 주고받는지**를 한곳에 정리했습니다. 처음 보는 사람이 이 문서만 보고 원하는 코드를 찾아갈 수 있게 하는 것이 목표입니다.

## 0. 문서 지도

| 알고 싶은 것 | 문서 |
|---|---|
| 설치, 시뮬·실물 실행 순서, 경기 규칙 | [README.md](../README.md) |
| 전체 구조 그림, 데이터 흐름, 공부 순서 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **파일별 역할·토픽·설정 (이 문서)** | CODE_GUIDE.md |
| 강화학습 설계, 학습 순서, 결과 기록 | [RL_TRAINING.md](RL_TRAINING.md) |

## 1. 폴더 한눈에 보기

```
pinky_soccer_ws/
├─ README.md                         설치·실행 안내
├─ docs/                             설명 문서 (이 폴더)
├─ config/                           CycloneDDS 설정 예시 (PC용 / 로봇용)
├─ scripts/                          실행 도우미 스크립트 (PC에서 실행)
│   └─ robot/                        로봇(라즈베리파이)에 복사해서 쓰는 스크립트
└─ src/
    ├─ pinky_soccer_bringup/         실행 묶음(launch): 시뮬 경기, 실물 경기, 단계별 실행
    ├─ pinky_soccer_game/            경기 두뇌: 공격수, 심판, 경기 창, 센서 창, 도메인 중계
    ├─ pinky_soccer_vision/          카메라: 공·라인 인식, 보정 창, YOLO, 로봇용 화면 프로그램
    ├─ pinky_soccer_nav/             위치 추정(AMCL)·주행(Nav2) 설정
    ├─ pinky_soccer_sim/             경기장 생성, MuJoCo 시뮬레이터, Gazebo 런치
    ├─ pinky_soccer_description/     로봇 모델 래퍼 + 앞 범퍼 (CAD 포함)
    ├─ pinky_soccer_rl/              강화학습: 학습 환경, 학습·재생 스크립트
    └─ pinky_pro/                    공식 핑키 레포 (저장소에 없음: setup 스크립트가 받아 옴)
```

로봇 쪽 드라이버(`robot.launch.py`, `command_watchdog` 등)는 이 저장소가 아니라 [`cse1598/pinky-soccer`](https://github.com/cse1598/pinky-soccer)의 로봇용 패키지입니다(README 3장).

## 2. 실행 진입점

### launch (자주 쓰는 것)
| 명령 | 하는 일 |
|---|---|
| `ros2 launch pinky_soccer_bringup sim_duel.launch.py` | MuJoCo 시뮬에서 로봇 2대 경기. 실물과 같은 구성(도메인 37/38) |
| `ros2 launch pinky_soccer_bringup real_duel.launch.py` | 실물 2대 경기 (PC 쪽 전부) |
| `ros2 launch pinky_soccer_bringup real_yellow.launch.py` | 실물 노랑 1대만 골 넣기 |
| `ros2 launch pinky_soccer_bringup sim_yellow.launch.py` | 시뮬 노랑 1대 (실물과 같은 설정) |
| `stage1_robot` ~ `stage6_game.launch.py` | 처음 개발할 때 쓰던 Gazebo 단계별 실행 (README 부록 A) |

### ros2 run (노드 하나씩)
| 패키지 | 실행 이름 | 파일 |
|---|---|---|
| pinky_soccer_game | `striker` | 공격수 (규칙 기반 상태 기계) |
| | `referee` | 시뮬 심판 (시뮬레이터 정답 위치 사용) |
| | `real_referee` | 실물 심판 (로봇 카메라의 공 위치 사용) |
| | `match_ui` | 경기 창 (점수판, 버튼) |
| | `sensor_view` | 센서 창 (두 로봇의 카메라·라이다·상태) |
| | `domain_relay` | 도메인 37 ↔ 38 메시지 중계 |
| | `opponent_bot` | Gazebo용 단순 상대 로봇 (예전 단계별 실행) |
| pinky_soccer_vision | `ball_detector` | 공·흰 라인 인식 |
| | `camera_calib` | 보정 창 (공 색, 렌즈, 카메라 각도, 바퀴) |
| | `collect_dataset`, `train_yolo` | YOLO 학습 데이터 수집과 학습 |
| pinky_soccer_sim | `mujoco_soccer` | MuJoCo 시뮬레이터 (sim_duel이 실행) |
| | `generate_world`, `generate_map` | 경기장 SDF / 점유 지도 생성 |

### scripts/ (PC)
| 스크립트 | 하는 일 |
|---|---|
| `setup_workspace.sh` | 처음 설치: 패키지 설치, 공식 레포 받기, 빌드 |
| `camera_calib.sh blue` | 실물 로봇 보정 창 (CycloneDDS 설정 포함) |
| `real_goal.sh` | 실물 노랑 1대 자동 실행 |
| `check_sim.sh` | Gazebo 시뮬 토픽·TF·Nav2 점검 |
| `rl_train.sh 1 3000000` | 강화학습 학습 |
| `rl_play.sh` | 학습 중인 정책을 MuJoCo 창에서 보기 |
| `apply_wheel_calib.sh yellow pinky@<IP>` | 보정 파일의 바퀴 반지름·간격을 로봇의 `~/start_robot.sh`에 넣고 드라이버를 다시 켬 |

### scripts/robot/ (로봇에 복사해서 사용)
| 스크립트 | 하는 일 |
|---|---|
| `start_robot.sh` | 로봇 드라이버 실행. `TEAM`, `WHEEL_RADIUS`, `WHEEL_SEPARATION` 설정 |
| `start_display.sh` | 팀 색 LED, LCD 국기·골 사진, 골 소리 (`pinky_display.py`) |
| `stop_soccer.sh` | 위 프로그램과 드라이버 노드 모두 종료 |

## 3. 패키지별 파일 설명

### 3-1. pinky_soccer_game: 경기 두뇌

#### `striker.py`: 공격수 (약 1000줄, 가장 중요)
- **하는 일:** 공 위치, 내 위치, 라이다를 보고 속도 명령을 냅니다. 상태 기계로 동작하며, 현재 상태는 `game/state`로 알립니다.
- **받는 것:** `ball/position_map`, `scan`, `line/boundary`, TF(map→base_footprint), `/game/control`, `/game/goal`, `/game/kickoff`, `/clicked_point`(RViz에서 목표 지정)
- **내는 것:** `cmd_vel_nav`(→ velocity_smoother → `cmd_vel`), `game/state`, `opponent/position_map`, `initialpose`(위치 추정이 경기장 밖으로 나갔을 때 다시 잡기), Nav2 `navigate_to_pose` 액션

**경기 진행 상태**
| 상태 | 뜻 |
|---|---|
| WAIT | 경기 전 / 정지 |
| PREPARING → READY | 시작 위치로 이동 → 도착 (12 cm 이상 밀리면 다시 정렬) |
| PREPARE_DETOUR / PREPARE_BACKOFF | 시작 위치로 가는 길에 상대를 돌아감 / 물러남 |
| CELEBRATE_GO → CELEBRATE_SPIN → CELEBRATE_DONE | 골 넣음: 자기 진영으로 → 3바퀴 → 완료 |
| CONCEDE_GO → CONCEDE_CENTRE | 골 먹음: 중앙선 근처로 비켰다가 시작 위치로 |
| SCORER_WAIT | 골 뒤 재시작에서 직전 득점 로봇이 3초 대기 |

**경기 중 상태**
| 상태 | 뜻 |
|---|---|
| SEARCH, LOOK_BACK, ROAM | 공 찾기: 기억한 위치 → 제자리 회전 → 뒤로 → 경기장 돌아다니기 |
| ORBIT, APPROACH | 공 뒤(골대 반대편)로 돌아가기 |
| DRIBBLE, SHOOT | 공 몰기, 슛 |
| OUT, NO_ROOM | 공이 골라인 밖 / 공 뒤로 갈 자리가 없음 (대기) |
| BACKOFF, ESCAPE | 막힘 → 물러나기, 반복해서 막히면 중앙 쪽으로 빠져나오기 |
| RL_PLAY | 강화학습 정책이 경기 중 판단을 맡는 중 (`policy_path` 설정 시) |
| RL_ESCAPE | RL 모드에서 라인 가드에 1.5초 넘게 막힘 → 1초 동안 경기장 중앙 쪽으로 |

- **안전장치:** 모든 속도 명령은 `GuardedPublisher`를 거쳐 두 가지 검사를 받습니다.
  - **라인 가드**(`_line_guard`): 흰 바깥 라인 쪽 이동을 막습니다.
  - **로봇 가드**(`_robot_guard`, `rl_policy.robot_guard`): 상대 로봇 쪽 이동을 32 cm부터 줄이고 20 cm에서 멈춥니다.
    - 공에 더 가까운 로봇은 우선권을 갖고 17 cm까지 갈 수 있고, 양보하는 로봇은 24 cm에서 멈춥니다.
    - 우선권은 `rl_policy.has_priority`로 정하고, 한 번 정해지면 6 cm 차이가 나기 전까지 바뀌지 않습니다. 직접 제어 명령은 상대 회피 필터(`avoidance.avoid_opponent`)와 골문 진입 방지(`goal_mouth_guard`)도 거칩니다.
- **주요 설정값:**

| 분류 | 설정값 |
|---|---|
| 속도 | `v_dribble`, `v_shoot`, `v_approach` |
| 공 뒤로 돌기 | `orbit_radius`, `behind_dist` |
| 슛 조건 | `shoot_range`, `shoot_lat`, `shoot_heading` |
| 세레모니 | `celebrate_turns` (3) |
| 경기 규칙 | `scorer_delay` (3초), `first_kick` (첫 출발 팀, 기본 노랑), `concede_centre`, `concede_hold` |
| 강화학습 | `policy_path` (정책 파일), `rl_search_after` (공을 이 시간 못 보면 규칙 기반 공 찾기, 3초) |
| 안전 | `robot_guard` (로봇 가드 켜기) |
| 막힘 처리 | `stuck_time`, `jam_dist`, `backoff_dist` |
| 라인 가드 | `line_guard`, `robot_front`, `robot_back`, `robot_half_width`, `line_margin` |

  시뮬·실물에서 쓰는 값은 `pinky_soccer_bringup/common.py`의 `solo_striker_params`에 모여 있습니다.

#### `rl_policy.py`: 강화학습 정책 실행
- `MLPPolicy`는 내보낸 `.npz`를 numpy로 계산합니다.
- `RLPlayer`는 공격수가 가진 정보로 학습 환경과 같은 17개 입력을 만들고, 행동을 (직진, 회전) 속도로 바꿉니다.
- 노랑·파랑 모두 공격 방향 기준 좌표로 계산합니다.
- `striker`에 `policy_path`(launch에서는 `policy_yellow:=`, `policy_blue:=`)를 주면 PLAY를 맡습니다(상태 `RL_PLAY`).

#### `avoidance.py`: 상대 피하기 도구
- `avoid_opponent`: 앞에 상대가 있으면 감속하고 옆으로 비켜 돕니다.
- `goal_mouth_guard`: 로봇 앞부분이 골대 안으로 들어가지 않게 합니다.
- `PrepareRouter`: 시작 위치로 갈 때 상대와 엉키지 않게 경유점을 거치거나 물러납니다.

#### `referee.py`: 시뮬 심판
- **하는 일:** 시뮬레이터의 정답 위치(`/gz/dynamic_pose`)로 골, 코너, 끼임을 판정하고 경기를 진행합니다.
- **경기 진행:** NEW_GAME → PREPARE → START → GOAL → … → FINISHED
- **시작 조건:** 두 공격수가 READY를 보내고, 정답 위치로도 시작 위치에 있어야 합니다.
- **공 처리:**
  - 골대 옆으로 나간 공은 센터로 옮깁니다.
  - 라인에 붙어 10초 동안 안 움직이는 공은 센터로 옮깁니다.
  - 모서리에 5초 끼인 공은 센터로 옮깁니다.
  - 랜덤 위치는 쓰지 않습니다.
- **로봇 구출:** 골대에 3초 끼인 로봇은 자기 시작 위치로 옮기고(3 cm 높이에서 내려놓음), 위치 추정을 다시 잡아 줍니다.
- **내는 것:** `/game/phase`, `/game/score`, `/game/event`, `/game/goals`, `/game/goal`, `/game/kickoff`, `/game/control`, `/sim/set_pose`, `/yellow/initialpose`, `/blue/initialpose`

#### `real_referee.py`: 실물 심판
- **하는 일:** 두 로봇 카메라가 보낸 공 위치로 판정합니다. 토픽은 시뮬 심판과 같아서 경기 창과 공격수는 그대로 동작합니다.
- **골 판정:** 0.6초 안에 골 안쪽 추정이 4번 이상 나와야 골입니다. 다른 로봇 카메라가 공을 멀리서 보고 있으면 무시합니다.
- **자동 시작:**
  - 두 로봇 READY + 공이 센터 12 cm 이내 + 2초 동안 가만히 있으면 시작합니다.
  - "가만히"는 카메라마다 따로 판단합니다. 두 카메라가 보는 위치가 몇 cm씩 다르기 때문입니다.
  - 기다리는 이유를 `자동 시작 대기: …`로 경기 창에 알립니다.
- **공 처리:** 골라인 밖 공, 라인에 붙은 공, 코너 공은 사람이 옮기도록 안내하고 소리를 냅니다.

#### `match_ui.py`: 경기 창
- 점수판, 골 기록, 버튼(경기 시작, 원위치, 직전 골 취소, 정지, 바로 시작, **공 가운데로**)이 있습니다. 버튼을 누르면 `/game/control`로 보냅니다.

#### `sensor_view.py`: 센서 창
- 로봇마다 카메라(공 인식 표시), 라이다(위에서 본 그림), 상태, 속도, 위치, 공 거리를 보여 줍니다.
- `--duel`은 두 도메인에 동시에 붙고, `--plain`은 실물(네임스페이스 없음)용입니다.

#### `domain_relay.py`: 도메인 중계
- 노랑(37) 도메인의 `/game/*`를 파랑(38)으로 복사합니다.
- 파랑의 `game/state`와 `ball/position_map`은 37의 `/blue/...`로 가져옵니다.
- 시뮬 심판이 보내는 `/blue/initialpose`도 넘깁니다.

#### `opponent_bot.py`
- Gazebo 단계별 실행에서 쓰던 단순 상대입니다(attacker, keeper, chaser, patrol, idle).

### 3-2. pinky_soccer_vision: 카메라

#### `ball_detector.py`: 공과 흰 라인 인식
- **공 인식:**
  1. 주황색(HSV) 덩어리를 찾습니다.
  2. 렌즈 왜곡을 보정하고, 카메라 각도(`cam_pitch_deg`)와 TF로 광선을 만듭니다.
  3. 광선이 공 중심 높이 평면과 만나는 점을 공 위치로 봅니다.
  4. `ball/position_base`(로봇 기준)와 `ball/position_map`(맵 기준)으로 보냅니다.
- **흰 라인 인식:**
  1. 흰 픽셀을 바닥으로 투영합니다.
  2. 맵 기준 경기장 가장자리 7 cm 안에 있으면 바깥 라인으로 봅니다. 골문 앞은 제외합니다.
  3. 바로 앞은 카메라 사각지대라 odom 기준으로 4초 기억합니다.
  4. `line/boundary`(로봇 기준 점들)로 보냅니다.
- **디버그 화면:** `ball/detection_image`에 공 상자, 빨간 점(바깥 라인), 초록 점(다른 라인)을 표시합니다.
- **보정값:** `real_duel`이 `~/.config/pinky_soccer/camera_<팀>.yaml`에서 읽어 넣습니다(intrinsics, distortion, cam_pitch_deg, HSV).
- `backend:=yolo`는 YOLO 모델, `backend:=gt`는 시뮬 정답(디버그용)을 씁니다.

#### `camera_calib_ui.py`: 보정 창
| 번호 | 보정 | 방법 |
|---|---|---|
| 1 | 공 색 (HSV) | 슬라이더 + 마스크 보기 |
| 2 | 카메라 각도 | 거리를 아는 공으로 측정·계산 (이분법) |
| 4 | 렌즈 | 체커보드 안쪽 코너 수집 → `calibrateCamera` |
| 5 | 바퀴 | 1 m 전진 → 반지름, 1바퀴 회전(IMU 비교) → 간격 |

- 결과는 **저장** 버튼을 누르면 `~/.config/pinky_soccer/camera_<팀>.yaml`에 기록됩니다.
- 카메라 화면 위에 공 위치(front/side/dist)를 크게 표시합니다.

#### `robot/` (로봇에서 실행)
- `pinky_display.py`: 팀 색 LED, LCD 국기와 골 사진, 골 멜로디. 자기 팀 `CELEBRATE*` 상태일 때 사진과 소리를 냅니다.
- `pi_camera_publisher.py`: 공식 bringup에 카메라 토픽이 없을 때 쓰는 카메라 노드(예전 1대 실행용)
- `pinky_effects.py`: LED 서비스(`led_server`)를 이용하는 예전 효과 노드

#### `collect_dataset.py`, `train_yolo.py`
- Gazebo에서 정답 위치로 YOLO 라벨을 자동으로 만들고 학습합니다(선택 기능).

### 3-3. pinky_soccer_sim: 경기장과 시뮬레이터

| 파일 | 하는 일 |
|---|---|
| `field.py` | 경기장 치수 읽기(`config/field_<style>.yaml`), Gazebo 월드 SDF와 점유 지도 생성. `resolve_dims`, `default_dims` |
| `rcj_field.py` | 경기장 박스·라인 목록 (cse1598/pinky-soccer에서 가져옴). MuJoCo와 Gazebo가 같이 씀 |
| `mujoco_soccer.py` | MuJoCo 시뮬레이터 (아래 표) |
| `urdf_tools.py` | 공식 모델 xacro를 팀별·카메라 각도별 URDF로 바꿈 (`plain_frames` 옵션) |
| `gz_utils.py` | Gazebo set_pose, 각도 계산 도우미 (`wrap`, `yaw_from_quat`) |
| `generate_world.py`, `generate_map.py` | 경기장 SDF / 지도 파일 생성 명령 |
| `launch/sim.launch.py` | Gazebo 시뮬 (예전 단계별 실행) |
| `media/` | LCD 그림: 국기 2장 (골 사진은 저작권 때문에 저장소에 없음) |
| `config/field_hardboard.yaml` | 훈련 경기장 치수: 1.66 × 1.06 m, 골 0.40 m, 바깥 여유 0.08 m |

**`mujoco_soccer.py` 상세**
| 부분 | 내용 |
|---|---|
| `build_scene` | 공식 pinky_mujoco 모델 2대 + 경기장 + 공을 하나의 장면으로 만듦. 팀 색 LED, LCD 화면, 라이다 머리 충돌 모양 포함. `markings`, `visuals`, `hull_verts` 옵션은 강화학습용 가벼운 모델 |
| `Robot` | 로봇 1대: 바퀴 속도 제어(가속 한계), 엔코더 거리 + IMU 방향 오도메트리, 라이다(광선), 카메라 렌더링, ROS 토픽 |
| `SoccerSim` | 도메인별 ROS 컨텍스트, `/clock`, 정답 위치(`/gz/dynamic_pose`), 위치 옮기기(`/sim/set_pose`), 세레모니 때 LCD 그림 바꾸기 |

### 3-4. pinky_soccer_nav: 위치 추정과 주행
- `launch/nav.launch.py`: 공식 Nav2 런치를 감싸고 다음을 준비합니다.
  - 경기장 치수로 만든 지도
  - AMCL 초기 위치
  - 네임스페이스(시뮬 2대)
  - `real_robot` 설정: 실물 라이다 잡음용 AMCL 값
  - `composition` (도메인마다 따로 프로세스)
- `config/nav2_params.yaml`: AMCL, 경로 계획, RPP 컨트롤러, velocity_smoother 설정
- `behavior_trees/soccer_navigate_to_pose.xml`: 공 옆에서 제자리 회전 복구를 뺀 행동 트리
- `maps/`, `rviz/`: 기본 지도와 RViz 설정

### 3-5. pinky_soccer_bringup: 실행 묶음
- `common.py`: 모든 launch가 같이 쓰는 경기장 인자와 공격수 설정(`solo_striker_params`)
- `launch/sim_duel.launch.py`:
  - 시뮬레이터(MuJoCo 또는 Gazebo)
  - 로봇마다(도메인 37/38): RSP, Nav2, 공·라인 인식, 공격수
  - 공용: 심판, 경기 창, 중계, 센서 창
- `launch/real_duel.launch.py`:
  - 로봇마다: Nav2, 공·라인 인식(보정 파일 적용), 공격수
  - 공용: 실물 심판, 경기 창, 중계, 센서 창
  - CycloneDDS 설정 포함

### 3-6. pinky_soccer_description: 로봇 모델
- `urdf/pinky_soccer.urdf.xacro`: 공식 모델에 앞 범퍼를 붙인 래퍼
- `urdf/bumper.urdf.xacro`, `config/bumper_v1_params.yaml`: ㄷ자 범퍼
- `cad/bumper/`: 범퍼 생성 스크립트(Blender), 3D 프린트용 STL

### 3-7. pinky_soccer_rl: 강화학습
| 파일 | 하는 일 |
|---|---|
| `env.py` | Gymnasium 학습 환경 (MuJoCo 직접 사용, ROS 없음). 입력 17개, 행동 2개, 보상·감점, 단계(1: 상대 없음, 2: 규칙 상대) |
| `train.py` | PPO 학습 (8개 병렬). 10만 스텝마다 평가해서 `best`/`latest` 저장, `eval.csv` 기록 |
| `play.py` | 정책을 MuJoCo 창에서 재생. `--follow`는 학습 중 최신 정책 자동 교체, `--scripted`는 비교용 단순 컨트롤러 |
| `export.py` | 학습한 정책을 ROS 공격수용 numpy 파일(`.npz`)로 내보내기. stable-baselines3와 결과 비교 검사 포함 |

자세한 내용은 [RL_TRAINING.md](RL_TRAINING.md)에 있습니다.

## 4. 토픽 정리

### 경기 공용 (`/game/*`, 노랑 도메인에서 나와 파랑 도메인으로 중계)
| 토픽 | 내용 | 보내는 쪽 |
|---|---|---|
| `/game/control` | `NEW_GAME`, `RESET`, `UNDO_GOAL`, `STOP`, `FORCE_START`, `BALL_CENTRE` (버튼) / `PREPARE`, `START`, `GOAL`, `WAIT` (심판 → 로봇) | 경기 창, 심판 |
| `/game/phase` | WAIT / PREPARE / START / GOAL / CORNER / FINISHED | 심판 |
| `/game/score`, `/game/goals`, `/game/event` | 점수, 골 기록, 안내 문장 | 심판 |
| `/game/goal` | 골 넣은 팀 (세레모니할 로봇) | 심판 |
| `/game/kickoff` | `OPEN` (선공 없음) | 심판 |
| `/game/fx` | 소리 효과: `GOAL`, `CORNER`, `FINISHED` | 심판 |
| `game/state` | 각 공격수의 현재 상태 (파랑은 `/blue/game/state`로 중계) | 공격수 |

### 로봇마다 (실물은 이름 그대로, 시뮬은 `/yellow/…`, `/blue/…`)
| 토픽 | 내용 |
|---|---|
| `camera/image_raw(/compressed)`, `scan`, `odom`, `imu`, `tf` | 센서 (로봇 또는 시뮬레이터) |
| `ball/position_map`, `ball/position_base`, `ball/detection_image` | 공 인식 결과 |
| `line/boundary` | 바깥 라인 점들 (로봇 기준) |
| `cmd_vel_nav` → `cmd_vel` | 공격수·Nav2 명령 → velocity_smoother → 로봇 (실물: watchdog → `/drive/cmd_vel`) |
| `amcl_pose`, `initialpose` | 위치 추정 결과 / 다시 잡기 |

### 시뮬 전용
| 토픽 | 내용 |
|---|---|
| `/gz/dynamic_pose` | 공·로봇 정답 위치 (심판용) |
| `/sim/set_pose` | 공·로봇 옮기기 |
| `/clock` | 시뮬레이션 시간 |

## 5. 설정 파일 위치 (저장소 밖)

| 위치 | 내용 |
|---|---|
| PC `~/.config/pinky_soccer/camera_<팀>.yaml` | 보정 결과: HSV, 카메라 각도, 렌즈, 바퀴 |
| PC `~/cyclonedds_pc.xml` | PC 통신 설정: 두 로봇 주소 (`config/cyclonedds_pc.xml.example`) |
| 로봇 `~/cyclonedds.xml` | 로봇 통신 설정: PC 주소 (`config/cyclonedds_robot.xml.example`) |
| 로봇 `~/start_robot.sh` 등 | 드라이버 실행 (`scripts/robot/`). 바퀴 반지름도 여기서 적용 |
| 로봇 `~/soccer_media/`, `~/pinky_display.py` | LCD 그림과 화면 프로그램 |
| PC `~/pinky_rl/stage<N>/` | 강화학습 결과: `best.zip`, `latest.zip`, `*_vecnorm.pkl`, `eval.csv`, `tb/` |
| PC `~/pinky_rl_venv/` | 강화학습용 파이썬 가상환경 (torch, stable-baselines3) |

## 6. 경기 하나가 진행되는 순서 (실물 기준)

1. 로봇: `start_robot.sh`가 센서와 바퀴 드라이버를 켭니다.
2. PC: `real_duel`이 로봇마다 Nav2, 공·라인 인식, 공격수를 켜고, 공용 심판, 경기 창, 중계를 켭니다.
3. **경기 시작** 버튼 → 심판이 `PREPARE` → 공격수가 시작 위치로 가서 `READY`
4. 공을 센터에 놓습니다 → 심판이 두 READY와 센터에 가만히 있는 공을 확인 → `START`
5. 경기: 공격수가 공을 찾고, 뒤로 돌아, 몰고, 슛합니다. 흰 라인은 넘지 않습니다.
6. 골 판정 → `GOAL` + `/game/goal`
   - 넣은 쪽: 세레모니 3바퀴, LCD 사진, 멜로디
   - 먹은 쪽: 중앙선 근처로 비켰다가 시작 위치로
7. 세레모니가 끝나면 `PREPARE` → 공을 센터에 놓음 → `START`. 직전 득점 로봇은 3초 늦게 출발합니다.
8. 먼저 5골을 넣으면 `FINISHED`

## 7. 변경 이력 요약

| 커밋 | 내용 |
|---|---|
| `faca130` | 첫 공개: MuJoCo 2대 시뮬, 실물 2대 경기, 보정 창, 문서 |
| `1c8d94c` | 카메라 라인 가드, 자동 시작 수정(카메라별 정지 판단, READY 유지), 3바퀴 세레모니, 실점 로봇 비키기, 득점 로봇 3초 대기, 반복 끼임 탈출, 위치 추정 다시 잡기, 시뮬 구출 수정 |
| (다음) | 강화학습 패키지 `pinky_soccer_rl`, 학습용 가벼운 모델 옵션, 문서 (CODE_GUIDE, RL_TRAINING) |
