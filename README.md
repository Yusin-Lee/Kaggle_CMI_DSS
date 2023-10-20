# Kaggle_CMI_DSS
Child Mind Institute - Detect Sleep States

과제 목표 : 주어진 anglez와 enmo를 기반으로 잠든 시각(onset)과 깨어난 시각(wakeup)을 찾아내는 것

기반 지식 from CMI Overview & EDA
1. anglez, enmo는 5초마다 측정
2. 270여 개의 train set 중에서 측정 기간 내 events가 빠진 Id를 제거하여 35개의 데이터 셋을 사용(refer - https://www.kaggle.com/code/carlmcbrideellis/zzzs-make-small-starter-datasets-target)
3. 하루에 1번의 onset과 wakeup만 존재. 다만, onset이 자정(00시)를 넘긴 시점에서는 하루에 2번의 onset이 존재할 수 있음. (ex, x월 27일 1시 onset -> x월 27일 8시 wakeup -> x월 27일 23시 onset)
4. 방식은 DL Base(Torch의 Conv1d베이스)와 ML Base(LGBM베이스) 나누어 시도
5. 태스크를 활동 상태와 수면 상태를 분류하는 classification으로 진행
6. (활동 상태 -> 수면 상태) 변화하는 시점을 onset으로, (수면 상태 -> 활동 상태) 변화하는 시점을 wakeup으로 간주
7. onset과 wakeup은 특정 시간에 주로 발생(ex, onset은 19시 ~ 03시, wakeup은 04시 ~ 11시)
8. 추가할 내용 : 35개의 데이터 셋을 기반으로 학습한 모델로 나머지 270여개의 데이터셋을 수도 라벨링

1) ML base

전처리 방식
- anglez, enmo를 standard normalize (mean과 std는 전체 train set에서 계산)
- 매 time step마다 이전 N분, 이후 N분 내의 변수의 std와 mean을 계산 [N = 60, 360, 720] -> [5m, 30m, 60m(1h)]
- 추가할 내용 : Lids(refer - https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/444007) -> 효과 ?

후처리 방식
- 수면 상태 및 활동 상태는 time step 주변 N분 동안은 동일한 상태일 가능성이 높다는 가정 하에 매 time step의 상태를 마다 30분 내의 가장 많이 예측된 상태로 변환
- 0시 ~ 6시 사이의 onset은 이전 일자의 onset으로 취급
- onset과 wakeup은 매일 한 번 발생하므로 가장 늦게 발생한 onset과 가장 먼저 발생한 wakeup을 선택
- @ 2023 10 19 추가내용 : onset과 wakeup이 발생한 시각이 1% 미만인 경우(in train set) 제외하고, 가장 먼저 발생한 onset과 가장 늦게 발생한 wakeup을 선택 -> 효과 O


2) DL base

전처리 방식
 - anglez, enmo를 standard normalize (mean과 std는 전체 train set에서 계산)
 - 매 time step의 이전 3분, 이후 3분의 anglez, enmo를 1d로 묶어 사용 -> [B, C, L]  == [B, 2(anglez, enmo), 72]

학습 방식
 - Conv1d 기반의 네트워크 사용(refer - )

후처리 방식
 - ML과 동일하게 처리
