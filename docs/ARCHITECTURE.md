# 구성 설명 · 공부 가이드

이 문서는 코드를 처음 읽는 사람을 위한 안내입니다. 큰 그림부터 시작해서 실제 파일로 내려갑니다.

## 1. 역할 나누기: 로봇은 몸, PC는 머리

| | 하는 일 |
|---|---|
| 로봇 (라즈베리파이) | 센서 값 보내기(라이다 `/scan`, 카메라 `/camera/image_raw/compressed`, 바퀴 `/odom`, IMU), 받은 속도 명령(`/cmd_vel`)대로 바퀴 돌리기, LED·LCD·부저 |
| PC | 위치 추정, 공 찾기, 판단, 경로 계산, 심판, 화면 |

라즈베리파이는 Nav2와 영상 처리를 같이 돌리기엔 느려서, 무거운 계산은 모두 PC가 합니다.

## 2. ROS 도메인으로 두 로봇 분리

두 로봇은 같은 토픽 이름(`/cmd_vel`, `/scan` …)을 씁니다. 같은 통신망에 있으면 노랑의 명령이 파랑에게도 가게 됩니다. 그래서:

- 노랑 = `ROS_DOMAIN_ID=37`, 파랑 = `ROS_DOMAIN_ID=38`. 도메인이 다르면 서로 보이지 않습니다.
- PC에서 노랑용 프로그램은 37번, 파랑용은 38번 도메인으로 켭니다([`real_duel.launch.py`](../src/pinky_soccer_bringup/launch/real_duel.launch.py)의 `robot('yellow', 37)`, `robot('blue', 38)`).
- 심판과 경기 창은 37번에 하나만 있습니다. [`domain_relay.py`](../src/pinky_soccer_game/pinky_soccer_game/domain_relay.py)가 한 프로세스 안에 도메인마다 하나씩 rclpy Context를 만들어 `/game/*` 메시지를 38번으로 복사하고, 파랑의 상태와 공 위치는 37번의 `/blue/...`로 가져옵니다.
- 통신은 CycloneDDS를 씁니다. 공용 와이파이는 멀티캐스트(자동 발견)를 막는 경우가 많아서, 설정 파일에 상대 주소를 직접 적습니다(`config/*.example`).

## 3. 한 로봇의 데이터 흐름

```
카메라 ──► ball_detector ──► /ball/position_map (맵 좌표 공 위치)
라이다 ──► AMCL ──► map→odom (맵에서 내 위치)
바퀴·IMU ──► /odom (얼마나 움직였나)
                   │
                   ▼
            striker (판단) ──► Nav2 목표 / 직접 속도 ──► /cmd_vel ──► 로봇
```

### 3-1. 공 검출 — [`ball_detector.py`](../src/pinky_soccer_vision/pinky_soccer_vision/ball_detector.py)
1. JPEG 영상을 풀고 HSV로 바꿔 주황 범위(`hsv_h_min` …)만 남깁니다.
2. 둥근 덩어리 중 가장 그럴듯한 것을 공으로 고릅니다.
3. 공 중심 픽셀에서 렌즈 왜곡을 없앱니다(`undistortPoints`, 체커보드 보정 값).
4. 카메라 내부값(fx, fy, cx, cy)으로 픽셀을 3D 방향(광선)으로 바꿉니다.
5. TF로 카메라가 로봇의 어디에, 몇 도로 달려 있는지 알고, `cam_pitch_deg` 보정을 더합니다.
6. 그 광선이 "공 중심 높이(2.1 cm) 평면"과 만나는 점이 로봇 기준 공 위치입니다. 깊이 카메라 없이 거리를 구하는 방법입니다.
7. TF로 맵 좌표로 바꿔 `/ball/position_map`으로 보냅니다.

→ 카메라 각도가 1°만 틀려도 먼 공의 거리가 크게 틀립니다. 그래서 보정 창이 필요합니다.

**흰 라인 (라인 가드):** 같은 노드가 흰 픽셀(채도 낮고 밝음)도 바닥으로 투영합니다. 맵 좌표로 바꿨을 때 경기장 가장자리에서 7 cm 안이면 바깥 라인으로 봅니다(골문 앞 제외). 바로 앞은 카메라 사각지대라 odom 좌표로 4초 기억해서 `/line/boundary`(로봇 기준 점들)로 보냅니다. striker는 모든 속도 명령을 내보내기 전에 이 점들이 로봇이 지나갈 띠 안에 있으면 직진·후진을 0으로 만듭니다(`_line_guard`).

### 3-2. 위치 추정 — AMCL ([`nav.launch.py`](../src/pinky_soccer_nav/launch/nav.launch.py))
- 경기장 맵은 치수 파일에서 자동으로 만듭니다(`pinky_soccer_sim/field.py`, `generate_map.py`).
- AMCL은 "내가 여기 있다면 라이다에 벽이 이렇게 보여야 한다"는 후보(파티클)를 많이 뿌리고, 실제 라이다와 잘 맞는 후보만 남기는 방식(파티클 필터)입니다.
- 바퀴 오도메트리로 후보들을 움직이고 라이다로 고칩니다. 바퀴 반지름이 틀리면 매번 고쳐야 하는 양이 커집니다(바퀴 보정이 필요한 이유).
- `real_robot:=true`에서는 실물 라이다 잡음에 맞춘 값(z_hit 0.9, sigma_hit 0.05 …)을 씁니다.

### 3-3. 판단 — [`striker.py`](../src/pinky_soccer_game/pinky_soccer_game/striker.py)
상태 기계(state machine)입니다. 지금 상태는 `/game/state`로 나옵니다.

| 상태 | 하는 일 |
|---|---|
| WAIT / PREPARE | 경기 전·재시작 전: 시작 위치로 이동 |
| SEARCH | 공이 안 보임: 기억한 위치 → 제자리 회전 → 뒤로 조금 → 경기장 여러 곳 돌아다니기(ROAM) |
| ORBIT / APPROACH | 공 "뒤쪽"(골대 반대편)으로 돌아가기. 공에 가까울수록 도는 반지름을 줄임 |
| DRIBBLE | 공을 앞면으로 밀며 조준점으로. 벽에 붙은 공은 벽을 따라 조준 |
| SHOOT | 골 가까이에서 정렬되면 짧게 돌진 |
| CELEBRATE | 골 뒤 자기 진영 가운데에서 3바퀴 회전 |
| CONCEDE | 실점 뒤: 중앙선 근처로 비켰다가 시작 위치로 |
| SCORER_WAIT | 골 뒤 재시작에서 직전 득점 로봇이 3초 대기 |

- 상대 회피: 라이다 점 중 맵의 벽으로 설명되지 않는 점을 상대로 보고, 가까우면 감속하며 옆으로 돕니다([`avoidance.py`](../src/pinky_soccer_game/pinky_soccer_game/avoidance.py)).
- 튜닝 값은 [`common.py`](../src/pinky_soccer_bringup/pinky_soccer_bringup/common.py)의 `solo_striker_params`에 모여 있습니다.

### 3-4. 주행 — Nav2
striker가 목표 자세를 주면 Nav2가 경로를 만들고(planner), 경로를 따라가는 속도를 계산하고(RPP controller), 속도를 부드럽게 다듬습니다(velocity_smoother). 몰기·슛·세레모니처럼 빠른 반응이 필요할 때는 striker가 직접 속도를 냅니다.

## 4. 경기 운영

- **심판** — [`real_referee.py`](../src/pinky_soccer_game/pinky_soccer_game/real_referee.py) (실물), [`referee.py`](../src/pinky_soccer_game/pinky_soccer_game/referee.py) (시뮬)
  - 실물 심판은 두 로봇 카메라의 공 위치를 받아 판정합니다. 한 번 튀는 값은 무시하고(0.6 m 이상 점프), 0.6초 안에 골 안에서 4번 보여야 골입니다.
  - 시뮬 심판은 시뮬레이터의 정답 위치(`/gz/dynamic_pose`)를 쓰고 공을 직접 옮길 수 있습니다(`/sim/set_pose`).
  - 진행 단계는 `/game/phase`, 점수는 `/game/score`, 화면·소리 효과는 `/game/fx`로 나옵니다.
- **경기 창** [`match_ui.py`](../src/pinky_soccer_game/pinky_soccer_game/match_ui.py): 버튼을 누르면 `/game/control`에 `NEW_GAME`, `STOP` 등을 보냅니다.
- **센서 창** [`sensor_view.py`](../src/pinky_soccer_game/pinky_soccer_game/sensor_view.py): 두 도메인에 동시에 붙어서 카메라·라이다·상태를 보여 줍니다.
- **로봇 표시** [`pinky_display.py`](../src/pinky_soccer_vision/robot/pinky_display.py): 로봇 안에서 `/game/state`가 `CELEBRATE…`가 되면 사진·LED·멜로디를 냅니다.

## 5. 시뮬레이션과 실물의 차이

판단 코드(striker, Nav2, ball_detector)는 같고, **센서 값을 어디서 받느냐**만 다릅니다.

| | 시뮬 (MuJoCo) | 실물 |
|---|---|---|
| 실행 | `sim_duel.launch.py` | 로봇: `robot.launch.py`, PC: `real_duel.launch.py` |
| 센서 | [`mujoco_soccer.py`](../src/pinky_soccer_sim/pinky_soccer_sim/mujoco_soccer.py)가 가상 카메라(오프스크린 렌더링)·라이다(광선 쏘기)·바퀴·IMU를 만들어 보냄 | 실제 장치 |
| 토픽 | `/yellow/…`, `/blue/…` (네임스페이스) | 이름 그대로, 도메인으로 구분 |
| 공 재배치 | 심판이 자동 | 사람이 |

MuJoCo 시뮬에서 공은 구름 마찰(condim 6)로 실제처럼 굴러가다 서고, 로봇 앞 LCD에는 텍스처를 바꿔 끼워 국기·사진을 보여 줍니다.

## 6. 보정

[`camera_calib_ui.py`](../src/pinky_soccer_vision/pinky_soccer_vision/camera_calib_ui.py) 한 창에서 합니다. 결과는 `~/.config/pinky_soccer/camera_<팀>.yaml`입니다.

| 값 | 원리 | 쓰는 곳 |
|---|---|---|
| HSV | 주황 범위 | ball_detector |
| 렌즈 (`intrinsics`, `distortion`) | 크기를 아는 체커보드를 여러 각도에서 찍어 OpenCV `calibrateCamera`로 계산 | ball_detector |
| `cam_pitch_deg` | 아는 거리에 공을 두고, 계산 거리가 실제와 같아지는 각도를 이분법으로 찾음 | ball_detector |
| `wheel_radius` | 오도메트리로 1 m 간 뒤 실제 거리를 재서 `새 반지름 = 지금 반지름 × 실제 / 오도메트리` | 로봇 드라이버 (`start_robot.sh`) |
| `wheel_separation` | 오도메트리로 360° 돈 뒤 IMU 각도와 비교 | 로봇 드라이버 |

## 7. 추천 공부 순서

1. **ROS 2 기본:** 노드, 토픽, 메시지, launch, TF, `ROS_DOMAIN_ID`. `ros2 topic list`, `ros2 topic echo /odom`, `ros2 run tf2_tools view_frames`를 직접 쳐 보세요.
2. **[`real_duel.launch.py`](../src/pinky_soccer_bringup/launch/real_duel.launch.py):** 무엇이 어느 도메인에서 켜지는지 지도처럼 봅니다.
3. **[`ball_detector.py`](../src/pinky_soccer_vision/pinky_soccer_vision/ball_detector.py):** 픽셀 → 바닥 좌표. 카메라 기하학(핀홀 모델) 공부에 좋습니다.
4. **[`striker.py`](../src/pinky_soccer_game/pinky_soccer_game/striker.py):** 상태가 언제 어떻게 바뀌는지 따라가 봅니다.
5. **[`real_referee.py`](../src/pinky_soccer_game/pinky_soccer_game/real_referee.py):** 경기 규칙을 코드로 어떻게 표현했는지 봅니다.
6. **AMCL·Nav2 개념:** 파티클 필터, 코스트맵, 경로 추종.
7. **[`mujoco_soccer.py`](../src/pinky_soccer_sim/pinky_soccer_sim/mujoco_soccer.py):** 시뮬레이터가 실물을 어떻게 흉내 내는지 봅니다.
