# 🌟 KOELECTRA-BiGRU

**KOELECTRA-BiGRU**는 **KOELECTRA** 모델과 **BiGRU**를 결합하여 **개체명 인식(NER)**, **속성명 분석**, **감성 분석**을 수행할 수 있는 한국어 자연어 처리(NLP) 프로젝트입니다.  
학습 데이터는 [AI허브 관광 특화 말뭉치 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71714)를 사용했으며, 학습된 모델과 실행 가능한 코드가 포함되어 있습니다.

---

## 🔑 **주요 특징**

- **기반 모델:** [KoELECTRA](https://github.com/monologg/KoELECTRA?tab=readme-ov-file)
- **파인튜닝 계층:** BiGRU (Bidirectional Gated Recurrent Unit)
- **주요 기능(NER):**
  - **개체명 인식:** 텍스트에서 특정 명사 인식
  - **속성명 인식:** 텍스트 내의 특정 속성 정보 인식
  - **감성명 인식:** 텍스트 감성(긍정, 부정, 중립) 인식
- **데이터셋:** AI허브 관광 특화 말뭉치
- **결과 파일:** 학습된 모델(`.pth`) 및 코드로 재현 가능

---

## 📂 프로젝트 구조

```plaintext
📦 프로젝트 디렉토리
├── test/                 # 검증 코드 및 관련 데이터
├── train/                # 학습 코드 및 관련 데이터
├── .gitattributes        # Git LFS 설정 파일
├── README.md             # 프로젝트 설명 파일
└── requirements.txt      # 프로젝트 의존성 패키지 목록
```

## ⚙️ **설치 및 실행**

### 1️⃣ 환경 설정

1. **Python 3.9 이상 설치**  
   Python이 설치되어 있지 않다면 [Python 공식 웹사이트](https://www.python.org/downloads/)에서 다운로드 및 설치하세요.

2. **필요한 패키지 설치**  
   아래 명령어로 프로젝트 실행에 필요한 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt

### 2️⃣ 모델 학습 및 실행

#### 💡 개체명 인식(NER)
- 학습 코드 실행:
  ```bash
  python src/object_train.py
#### 💡 속성명 인식(NER)
- 학습 코드 실행:
  ```bash
  python src/attribute_train.py
#### 💡 감성 인식(NER)
- 학습 코드 실행:
  ```bash
  python src/emotion_train.py
  ```
  
## 🚀 모델 학습

### KoELECTRA-BiGRU 기반 개체명 인식 모델
- **개체명 인식 모델 개발**을 위해 KoELECTRA-BiGRU를 결합하였으며, 모델 학습에는 **학습용 데이터(80%)**, **검증용 데이터(10%)**, **시험용 데이터(10%)**를 사용하였습니다.

### 학습/검증/시험 데이터 구성
| 단계       | 설명                                  | 데이터 구성      |
|------------|---------------------------------------|------------------|
| **학습**   | KoELECTRA-BiGRU 모델 학습             | 전체 데이터 80%  |
| **검증**   | 학습 도중 모델 성과 평가 및 비교      | 전체 데이터 10%  |
| **시험**   | 모델 학습 완료 후 최종 테스트         | 전체 데이터 10%  |

## 📊 **모델 성능**

| **클래스**          | **F1-스코어** |
|----------------------|---------------|
| **속성 (attribute)**   | **0.9996**    |
| **개체 (object)** | **0.8809**    |
| **감성 (emotion)**    | **0.8168**    |



### 🧩 모델 개발 단계
아래 그림은 모델 개발 단계를 설명합니다:

![NER 모델 개발 단계](images/arc.png)

---
## 📜 데이터셋 설명
이 프로젝트는 AI허브 관광 특화 말뭉치 데이터셋을 사용하여 학습하였습니다. 데이터셋은 관광지 정보, 감성 태그, 속성 태그 등으로 구성되어 있습니다.

데이터셋 구성 예시

```json
{
    "info": {
        "creator": "세명소프트",
        "description": "관광 특화 말뭉치"
    },
    "docu_info": {
        "content": "영천 목재문화체험장",
        "sentences": [
            {
                "sentenceId": "0001",
                "sentence": "관광지명 영천 목재문화체험장",
                "annotations": [
                    {
                        "TagText": "영천",
                        "TagCode": "LC",
                        "startPos": 5,
                        "endPos": 6
                    },
                    {
                        "TagText": "목재문화체험장",
                        "TagCode": "LC",
                        "startPos": 8,
                        "endPos": 14
                    }
                ]
            }
        ]
    }
}
```

## 🔖 **태그 정의**

### **개체명(NER)**

| 번호 | 분류   | 코드 | 설명                         |
|------|--------|------|------------------------------|
| 1    | 사람   | PS   | 실존 인물 및 가상 캐릭터 등  |
| 2    | 지역   | LC   | 장소를 뜻하는 표현           |
| 3    | 기관   | OG   | 기관 및 단체                |
| 4    | 인공물 | AF   | 사람에 의해 창조된 대상물    |
| 5    | 날짜   | DT   | 구체적인 날짜 표현           |
| 6    | 문명   | CV   | 문명/문화 관련 표현          |
| 7    | 동물   | AM   | 특정 동물                   |
| 8    | 식물   | PT   | 특정 식물                   |
| 9    | 수량   | QT   | 수량 표현                   |
| 10   | 사건   | EV   | 사회운동, 선언, 조약 등      |

---

### **속성명**

| 번호 | 분류     | 코드 | 설명                        |
|------|----------|------|-----------------------------|
| 1    | 주소     | AD   | 주소 정보                  |
| 2    | 교통     | TR   | 대중교통 및 길 안내         |
| 3    | 일정     | DA   | 특정 영업 날짜 및 행사 기간 |
| 4    | 용어     | TM   | 홈페이지, 이메일 주소 등    |
| 5    | 전화번호 | TE   | 안내 전화번호              |
| 6    | 우편번호 | PO   | 우편번호                  |
| 7    | 시간     | TI   | 운영 시간, 입퇴실 시간      |
| 8    | 가격     | PR   | 이용료 및 상품 가격         |
| 9    | 부대정보 | UN   | 상품/서비스 유형 정보       |
| 10   | 기타사항 | ET   | 그 외 정보                 |

---

### **감성명**

| 번호 | 분류   | 코드 | 설명                     |
|------|--------|------|--------------------------|
| 1    | 긍정   | P    | 긍정적 감성 표현         |
| 2    | 중립   | NA   | 긍정/부정이 아닌 표현     |
| 3    | 부정   | N    | 부정적 감성 표현         |

## 📜 인용
KoELECTRA 인용

```json
@misc{park2020koelectra,
  author = {Park, Jangwon},
  title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monologg/KoELECTRA}}
}
```

데이터셋 출처
데이터셋: AI허브 한국어 텍스트 데이터셋
