# MoodFlow

Spotify 오디오 Feature로 곡의 무드를 분류하고, Seed 곡 하나를 고르면 같은 분위기의 플레이리스트를 만들어 주는 프로젝트입니다.
추천만 하는 게 아니라 **재생 순서**까지 정해서 곡 사이 분위기 변화가 덜 튀도록 만드는 게 목표였습니다.

## 어떻게 만들었나

| 단계 | 내용 | 파일 |
|---|---|---|
| 전처리 | 곡명+아티스트 중복, 결측치, tempo 0, 50초 미만·400초 초과 곡 제거 (114,000곡 → 76,630곡) | `src/preprocess.py` |
| EDA | 사용할 Feature 6개의 분포 확인 | `notebooks/01_eda.ipynb` |
| 무드 라벨링 | Feature를 순위(0~1)로 바꾼 뒤 가중합으로 7개 무드(chill, focus, romantic, happy, sad, energetic, dreamy) 점수 계산, 가장 높은 무드로 지정 | `notebooks/02_mood_labeling.ipynb` |
| 군집화 | 무드별로 K-Means (K=3) | `notebooks/03_clustering.ipynb` |
| 추천 | Seed와 같은 무드·군집 안에서 Cosine 유사도 순으로 추천, 같은 아티스트는 최대 2곡 | `notebooks/04_recommendation.ipynb` |
| 평가 | 추천 방식, 순서 방식 비교 | `notebooks/05_evaluation.ipynb` |
| 순서 최적화 | Seed에서 시작해 Nearest Neighbor로 순서를 정하고 2-opt로 개선 | `src/recommend.py` |
| API | FastAPI | `app/backend/main.py` |

사용한 Feature: `danceability`, `energy`, `tempo`, `acousticness`, `instrumentalness`, `valence`

## 폴더 구조

```
.
├── app
│   ├── backend/main.py      FastAPI 서버
│   └── frontend/            React 화면 (작업 중)
├── data
│   ├── raw/dataset.csv      원본 (저장소에 포함하지 않음)
│   └── processed/           전처리·라벨링·군집화 결과 (저장소에 포함하지 않음)
├── models/                  scaler, K-Means (03 실행 시 생성)
├── notebooks/               01 ~ 05
├── reports/                 분석 결과 CSV
├── src
│   ├── preprocess.py
│   └── recommend.py
└── requirements.txt
```

## 실행 방법

1. 패키지 설치

   ```
   pip install -r requirements.txt
   ```

2. 데이터 준비: [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)의 `dataset.csv`를 `data/raw/`에 넣습니다. 폴더가 없으면 만들어야 합니다.

3. 전처리 (프로젝트 루트에서 실행)

   ```
   python src/preprocess.py
   ```

4. 노트북을 `01`부터 `05`까지 순서대로 실행합니다. 노트북은 `notebooks/` 폴더에서 열어야 경로가 맞습니다.
   서버만 띄울 거라면 `03`까지만 실행해도 됩니다.

5. 서버 실행 (프로젝트 루트에서 실행)

   ```
   python -m uvicorn app.backend.main:app --reload
   ```

   `http://127.0.0.1:8000/docs`에서 API를 바로 테스트할 수 있습니다.

## API

| 주소 | 설명 |
|---|---|
| `GET /moods` | 무드 목록과 곡 수 |
| `GET /seeds?mood=chill&limit=10` | 무드별 인기곡 (Seed 후보) |
| `GET /search?q=love&mood=&limit=20` | 곡명·아티스트 검색 |
| `GET /playlist?seed_id=...&size=20&method=nn_2opt&use_cluster=true` | Seed 곡으로 플레이리스트 생성 |

`/playlist` 응답에는 곡 목록(`tracks`), 순서 최적화 후 전환 비용(`total_cost`), 유사도순 비용(`similarity_order_cost`), 감소율(`cost_reduction_percent`), Spotify 재생기 주소(`embed_url`)가 들어 있습니다.

## 평가 결과

무드별로 Seed 15곡씩, 총 105곡으로 평가했습니다. 곡 사이 거리는 표준화한 Feature 기준 유클리드 거리이고, 플레이리스트는 Seed 포함 20곡입니다.

**추천 방식**

| 방식 | 평균 Cosine 유사도 |
|---|---|
| 무작위 (같은 군집) | 0.659 |
| Cosine (같은 무드) | 0.981 |
| Cosine (같은 군집) | 0.980 |

같은 무드 전체에서 찾을 때와 같은 군집에서 찾을 때의 차이가 거의 없어서, 군집 필터의 효과는 확인하지 못했습니다.

**재생 순서 (평균 전환 비용, 낮을수록 부드러움)**

| 순서 | 비용 | 무작위 대비 | 유사도순 대비 |
|---|---|---|---|
| 무작위 | 13.28 | - | - |
| 유사도순 | 12.92 | -2.7% | - |
| Nearest Neighbor | 7.89 | -40.6% | -38.9% |
| NN + 2-opt | 7.40 | -44.2% | -42.7% |

105개 Seed 모두에서 NN + 2-opt의 비용이 무작위, 유사도순보다 낮았습니다.

## 한계

- 무드는 사람이 붙인 라벨이 아니라 오디오 Feature로 만든 규칙 기반 라벨입니다. 같은 Feature로 라벨링과 추천을 하기 때문에 "무드 일치율"은 평가 지표로 쓰지 않았습니다.
- 추천된 곡이 장르는 다양하게 섞입니다(Seed 하나 기준 20곡이 17개 장르). Feature는 비슷해도 실제로 이어 들으면 어색할 수 있습니다.
- 사용자 평가는 아직 하지 않았습니다.

## 데이터 출처

Kaggle의 Spotify Tracks Dataset (maharshipandya, 114,000곡, 125개 장르)을 사용했습니다. 원본 CSV는 이 저장소에 포함하지 않았고, 사용 조건은 Kaggle 데이터셋 페이지를 따릅니다.
