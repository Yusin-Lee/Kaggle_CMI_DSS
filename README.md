# Kaggle_CMI_DSS
Child Mind Institute - Detect Sleep States

과제 목표 : 주어진 anglez와 enmo를 기반으로 잠든 시각(onset)과 깨어난 시각(wakeup)을 찾아내는 것

# 1. 기반 지식, 기초 전처리 from CMI Overview & EDA
   - anglez, enmo는 5초마다 측정
   - 270여 개의 train set 중에서 측정 기간 내 events에 결측이 있는 ID를 제거하여 35개의 데이터 셋을 사용(refer - https://www.kaggle.com/code/carlmcbrideellis/zzzs-make-small-starter-datasets-target)
   - @ 2023 11 01 추가 내용 : 결측이 있는 데이터는 결측을 제외한 날짜만 사용하여 trainset으로, 결측이 없는 35개의 데이터는 valid set으로 사용
   - @ 데이터 중 7개 이상의 {onset - wakeup} 페어가 있는 ID만 사용
   - 하루에 1번의 onset과 wakeup만 존재. 다만, onset이 자정(00시)를 넘긴 시점에서는 하루에 2번의 onset이 존재할 수 있음. (ex, x월 27일 1시 onset -> x월 27일 8시 wakeup -> x월 27일 23시 onset)
   - train dataset 내에 event는 모두 onset으로 시작하여 wakeup으로 끝
   - @ 2023 11 01 추가 내용 : event는 모두 {onset - wakeup} 페어로 이루어졌으나, 둘 중 하나가 Nan인 경우는 존재 -> pair group으로 묶어 둘 중 하나가 Nan경우는 제외
   - 방식은 DL Base(Torch의 Conv1d베이스)와 ML Base(LGBM베이스) 나누어 시도
   - 태스크를 활동 상태와 수면 상태를 분류하는 classification으로 진행
   - (활동 상태 -> 수면 상태) 변화하는 시점을 onset으로, (수면 상태 -> 활동 상태) 변화하는 시점을 wakeup으로 간주
   - onset과 wakeup은 특정 시간에 주로 발생(ex, onset은 19시 ~ 03시, wakeup은 04시 ~ 11시)

# 2. ML base
   - 전처리 방식
     - anglez, enmo를 standard normalize (mean과 std는 전체 train set에서 계산) -> 효과 X
     - 매 time step마다 이전 N-timestep 이전, 이후의 std와 mean을 계산 [N = 60, 360, 720] -> [5m, 30m, 60m(1h)]
     - @ 2023 10 19 추가내용 : Lids(refer - https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/444007) -> 성능 향상에는 의문부호이나 일단 사용
     - @ 2023 10 25 추가내용 : 매 time step마다 [-N/2 , + N/2]의 범위내의 std와 mean을 계산 -> 효과 X
     - @ 2023 10 31 추가내용 : Fast Fourier Transformation 적용(refer - https://www.kaggle.com/code/jjinho/cmi-sleep-detection-fast-fourier-transformation) / -> 효과 X
     - @ 2023 10 31 추가내용 : std/mean 계산에 3600(5h) 추가 -> 효과 X

   - 후처리 방식
     - 수면 상태 및 활동 상태는 time step 주변 N분 동안은 동일한 상태일 가능성이 높다는 가정 하에 매 time step의 상태를 마다 30분 내의 가장 많이 예측된 상태로 변환
     - 0시 ~ 6시 사이의 onset은 이전 일자의 onset으로 취급
     - onset과 wakeup은 매일 한 번 발생하므로 가장 늦게 발생한 onset과 가장 먼저 발생한 wakeup을 선택
     - @ 2023 10 19 추가내용 : onset과 wakeup이 발생한 시각이 1% 미만인 경우(in train set) 제외하고, 가장 먼저 발생한 onset과 가장 늦게 발생한 wakeup을 선택 -> 효과 O


# 3. DL base
   - 전처리 방식
     - 매 time step마다 이전 N분, 이후 N분 내의 변수의 std와 mean을 계산 [N = 60, 360, 720] -> [5m, 30m, 60m(1h)] -> 효과 X
     - @ 2023 10 15 추가내용 : 매 time step의 이전 3분, 이후 3분의 anglez, enmo, 이전에 만든 변수를 1d로 묶어 사용 -> [B, C, L]  == [B, 27(anglez, enmo, ...), 72] -> 효과 X
     - @ 2023 10 31 추가내용 : Fast Fourier Transformation(이하 FFT) 적용(refer - https://www.kaggle.com/code/jjinho/cmi-sleep-detection-fast-fourier-transformation) -> CV Score/Train loss에 효과 있는 것으로 보임, 더 확인해야됨
     - @ 2023 11 01 추가내용 : anglez는 normalize, enmo는 fft 적용 시 성능 상승이 가장 높은 것으로 보임

   - 후처리 방식
     - ML과 동일하게 처리
