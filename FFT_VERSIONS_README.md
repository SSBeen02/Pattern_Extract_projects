# FFT 버전 비교 및 crop 변수 안내

대상: `period_candidates_fft.py`, `period_candidates_fft2.py`

## 1. 두 버전의 차이

두 버전은 RGB 축별 평균 신호의 1D FFT로 기본 주기를 검출하고, 가로·세로 각각 1~4배인 중앙 패치 16개를 만든다. 2D FFT는 시각화용이며 최종 주기 선택에는 사용하지 않는다. 차이는 패치를 반복 확장하여 원본과 비교하는 영역의 크기다.

| 항목 | fft | fft2 |
|---|---|---|
| 추가하는 전체 가로·세로 길이 | 실제 패치의 너비·높이 | 반올림한 기본 주기, 각각 최소 1픽셀 |
| 확장 후 크기 | `2 * patch.width`, `2 * patch.height` | `patch.width + extra_w`, `patch.height + extra_h` |
| 배수 증가 시 패딩 | 패치 크기에 따라 증가 | 배수와 무관하게 일정 |
| 확장 함수 | `extend_patch_edges(patch)` | `extend_patch_edges(patch, extra_w, extra_h)` |

예를 들어 가로 기본 주기가 20픽셀이고 패치가 원본 크기로 제한되지 않는다면:

| 가로 배수 | 패치 너비 | fft 비교 너비 | fft2 비교 너비 |
|---|---:|---:|---:|
| 1 | 20 | 40 | 40 |
| 2 | 40 | 80 | 60 |
| 3 | 60 | 120 | 80 |
| 4 | 80 | 160 | 100 |

두 버전 모두 `np.pad(..., mode="wrap")`으로 반대쪽 가장자리를 반복 연결한다. 추가 길이가 홀수이면 오른쪽·아래쪽에 1픽셀을 더 배분한다. 확장 영역이 원본 밖으로 나가면 비교 이미지는 `None`, 점수는 `NaN`, 상태는 `outside_image`가 된다. 비교 범위가 달라지므로 두 버전의 NCC 점수와 순위도 달라질 수 있다.

## 2. crop 관련 변수

좌표와 크기는 픽셀 단위다. 이미지의 왼쪽 위가 원점이고, 오른쪽이 +x, 아래쪽이 +y다.

| 변수 | 역할 |
|---|---|
| `image` | 원본 RGB 이미지 |
| `base_dx`, `base_dy` | 검출한 가로·세로 기본 주기 |
| `dx_mult`, `dy_mult` | 기본 주기에 곱하는 배수, 각각 1~4 |
| `patch_w`, `patch_h` | 요청 패치 크기: `int(round(base_dx * dx_mult))`, `int(round(base_dy * dy_mult))` 여기서 이미지 사이즈에서 벗어나지 않도록 조정됨.|
| `left`, `top` | 기본 패치의 왼쪽·위쪽 경계 좌표 |
| `right`, `bottom` | 기본 패치의 오른쪽·아래쪽 경계 좌표. 해당 끝 경계는 crop에 포함되지 않음 |
| `image_patch` | 원본 중앙에서 잘라낸 기본 패치 |
| `image_patch.width`, `image_patch.height` | 실제 패치 크기. `center_patch_box`가 요청 크기를 원본 크기 이내, 최소 1픽셀로 제한 |
| `extra_w`, `extra_h` | fft2에서 추가할  가로·세로 길이: `max(1, round(base_dx))`, `max(1, round(base_dy))` |
| `box` | 패딩을 포함한 원본 비교 영역의 `(left, top, right, bottom)` 좌표 |
| `fits` | 비교 영역 전체가 원본 이미지 안에 들어가는지 여부 |
| `extended` | 기본 패치를 반복 확장한 이미지 |
| `reference` | 원본에서 `box`로 잘라낸 비교 이미지. `fits`가 거짓이면 `None` |
| `score`, `status` | NCC 점수와 비교 상태: `ok`, `zero_energy`, `outside_image` |
| `comparison_width`, `comparison_height` | 결과 메타데이터에 저장하는 확장 이미지 너비·높이 |



## 3. fft2에 정의된 함수

| 함수 | 역할 |
|---|---|
| `ensure_dir` | 출력 디렉터리가 없으면 생성 |
| `load_rgb` | 이미지를 RGB로 읽어 PIL 이미지와 float32 배열 반환 |
| `normalize_channel` | 평균 제거 후 표준편차로 정규화. 상수 신호는 나눗셈 생략 |
| `normalize_01` | 유한한 최솟값·최댓값을 기준으로 0~1 정규화 |
| `fft_rgb_log_magnitude` | RGB 채널별 2D FFT를 합쳐 시각화용 로그 크기 영상 생성 |
| `axis_period_profiles` | RGB 축별 평균 신호에 1D FFT를 적용해 진폭 프로파일과 축별 피크 반환 |
| `subpixel_peak_1d` | 인접한 세 샘플의 포물선 보간으로 피크 위치 보정량 계산 |
| `refine_profile_peak` | 피크 주파수 위치를 보정하고 소수점 한 자리로 반올림 |
| `select_axis_amplitude_peaks` | 유효 범위에서 정규화 진폭 0.73 이상인 국소 피크 중 가장 작은 k 선택. 없으면 유효 최댓값 선택 |
| `candidates_from_axis_period_peaks` | 축별 피크 주파수를 공간 주기 후보로 변환 |
| `basic_ncc` | 동일 크기 RGB 이미지의 평균 제거 없는 NCC 계산. 에너지가 없으면 NaN 반환 |
| `extend_patch_edges` | 지정한 추가 길이를 좌우·상하에 나눠 패치를 반복 확장 |
| `draw_ncc_montage` | 패치별 원본 비교 이미지, 반복 확장 이미지와 NCC를 몽타주로 저장 |
| `save_period_patches` | 16개 패치 생성, 확장·원본 비교, NCC 순위 및 CSV·미리보기·몽타주 저장 |
| `main` | CLI 인자 처리부터 이미지 로딩, 주기 검출, 결과 저장까지 전체 흐름 실행 |

외부 모듈 `period_candidates_visualization.py`에서 가져오는 함수는 다음과 같다.

| 함수 | 역할 |
|---|---|
| `draw_frequency_domain_peaks` | 주파수 영역의 피크 시각화 |
| `draw_spatial_domain_peaks` | 공간 영역의 주기 후보 시각화 |
| `save_axis_period_functions` | 축별 주기 프로파일과 피크 그래프 저장 |
| `save_patch_box_preview` | 원본 위에 패치 crop 영역 표시 |
| `safe_font` | 시각화에 사용할 폰트 로딩 |

## FFT2 패치 점수 선택

`period_candidates_fft2.py`에서 패치 생성, 마진 점수 계산, 순위 정렬을 수행하고,
`period_candidates_visualization.py`는 전달된 패치와 점수만 시각화합니다.

```powershell
python period_candidates_fft2.py --image input.png --output_dir outputs/ncc --score-method ncc
python period_candidates_fft2.py --image input.png --output_dir outputs/vgg19 --score-method vgg19
python run_all_period_patterns2.py --score-method vgg19
```

- 기본값 `ncc`: `opencv_ncc`로 채널별 평균을 제거한 pooled NCC를 계산합니다. 기존과 같이 확장 마진만 비교합니다. 상수 영역은 점수를 계산할 수 없어 N/A입니다.
- `vgg19`: RGB 패치를 원본 좌표에 맞춰 원본 크기(1024×1024)로 타일링하고, 원본과 동일한 VGG19 conv2_2 ReLU 출력 및 ToTensor 전처리를 사용합니다. 특징 맵 크기(512×512)에서 직접 boolean 마스크를 만들어 중앙 패치만 제외하고 나머지 전체를 비교합니다. 중앙 경계는 왼쪽/위 floor, 오른쪽/아래 ceil로 제외하며 마스크 축소나 부분 가중치는 없습니다. 양쪽 모두 반응이 0인 채널을 제외한 후, 비교 영역에서 양쪽 각각의 채널별 평균을 빼고 채널별 MSE를 평균합니다. MSE는 낮을수록 좋으며 오름차순으로 순위를 매깁니다. 비교 영역이 없거나 모든 채널이 제외되면 N/A입니다. 한쪽만 0인 채널도 MSE 계산에 포함합니다. VGG 전용 몽타주는 Patch / Tiled RGB / Original RGB를 표시하며 빨간 상자는 제외되는 중앙 패치입니다. NCC 마스크와 몽타주는 그대로 유지합니다.

- NCC는 점수가 높을수록 유사하며, 원본 범위를 벗어난 비교 영역은 N/A입니다.
- 시각화는 `draw_ncc_montage`와 `draw_vgg19_montage`로 분리되어 있고 파일명에도 `ncc_` / `vgg19_`가 붙습니다.
- `--pout-opt off`이면 패치 이미지 저장을 생략합니다. 패치 점수 계산은 수행합니다.

### RGB 다수결 피크 선택과 정규화

x/y 각각 RGB가 선택한 FFT bin을 기준으로 다수결 1등을 최종 피크로 사용합니다.
세 채널이 모두 다른 bin이면 선택된 피크의 amplitude가 가장 큰 채널을 사용합니다.
다수결로 일치한 채널 중 먼저 나온 채널의 소수점 보정값과 amplitude를 유지합니다.
NCC/VGG19 점수가 낮아도 피크를 바꾸거나 소수 후보를 재평가하지 않습니다.
패치 랭킹은 NCC 내림차순, VGG19 MSE 오름차순입니다.

FFT 입력은 RGB 각 채널별로 평균을 제거하고 표준편차로 나눠 정규화합니다.
표준편차가 1e-6 이하이면 나눗셈은 생략합니다(상수 채널은 0).
이후 축별 평균 제거, Hann window, FFT 절댓값 합산을 수행합니다.
FFT amplitude 자체에는 채널별 최대값 나누기나 0~1 정규화를 추가 적용하지 않습니다.
따라서 amplitude 비교는 표준화된 RGB 입력에서 나온 피크 크기를 비교합니다.
