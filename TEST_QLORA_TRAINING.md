# 🧪 QLoRA 자동 학습 테스트 가이드

## 📋 테스트 체크리스트

### 사전 준비

- [ ] PEFT 라이브러리 설치 확인
  ```bash
  pip show peft transformers accelerate
  ```

- [ ] GPU 사용 가능 확인 (선택)
  ```python
  import torch
  print(torch.cuda.is_available())  # True면 GPU 사용 가능
  ```

- [ ] 디스크 공간 확인 (최소 5GB 여유)

## 🎯 테스트 시나리오

### 시나리오 1: 회의 녹음 후 자동 학습

#### 1단계: 설정 확인
```
1. Persona-AI 실행
2. Settings 탭 이동
3. "QLoRA 페르소나 학습" 섹션 확인:
   - ☑ 회의 종료 시 자동 학습 (체크됨)
   - 최소 발언 수: 20
```

#### 2단계: 테스트 회의 진행
```
1. Live 탭 이동
2. "Start Recording" 클릭
3. 최소 20회 이상 발언 (예시):

   화자 A (15회):
   - "안녕하세요, 오늘 회의를 시작하겠습니다"
   - "첫 번째 안건은 데이터베이스 최적화입니다"
   - ... (13회 더)

   화자 B (25회):
   - "네, 말씀하신 부분에 동의합니다"
   - "저는 인덱스 추가를 제안드립니다"
   - ... (23회 더)

4. "Stop Recording" 클릭
```

#### 3단계: 학습 진행 확인
```
예상 동작:
1. 녹음 완료 메시지 표시
2. Status에 "회의 종료: 2명 참여자 기록 업데이트" 출력
3. 화자별 발언 수 체크:
   - ⏭ speaker_A: 발언 수 부족 (15/20) - 학습 건너뜀
   - 🧠 speaker_B QLoRA 학습 시작 (발언: 25개)
4. Live 탭 우측 하단에 진행 위젯 표시:
   - 📊 speaker_B 데이터셋 생성 중... (0-30%)
   - 🧠 speaker_B 말투 학습 중... (30-90%)
   - ✅ speaker_B 학습 완료! (100%)
5. 완료 알림창 표시
```

#### 4단계: 결과 확인
```bash
# 데이터셋 생성 확인
ls data/persona_datasets/
# 출력: speaker_B_dataset_YYYYMMDD_HHMMSS.jsonl

# 어댑터 생성 확인
ls adapters/speaker_B/final/
# 출력:
# - adapter_config.json
# - adapter_model.bin
# - metadata.json
# - tokenizer/
```

#### 5단계: 페르소나 사용 테스트
```
1. Persona Chatbot 도크 열기
2. 페르소나 드롭다운에서 "speaker_B" 선택
3. 질문 입력: "데이터베이스 최적화 방법을 알려주세요"
4. speaker_B의 말투로 답변이 오는지 확인
```

### 시나리오 2: 오프라인 파일 처리 후 자동 학습

#### 1단계: 오디오 파일 준비
```
테스트용 회의 녹음 파일:
- output/recordings/meeting_YYYYMMDD_HHMMSS.wav
또는 외부 파일 준비
```

#### 2단계: 파일 업로드
```
1. Minutes 탭 이동
2. "Load from File" 클릭
3. 오디오 파일 선택
4. 처리 완료 대기
```

#### 3단계: 자동 학습 확인
```
파일 처리 완료 후:
1. 자동으로 학습 트리거
2. 진행 상황 표시 (시나리오 1과 동일)
```

### 시나리오 3: 수동 설정 변경

#### 테스트 3-1: 자동 학습 끄기
```
1. Settings 탭 → "회의 종료 시 자동 학습" 체크 해제
2. 테스트 회의 진행 (20회 이상 발언)
3. Stop Recording 클릭
4. 학습이 시작되지 않는지 확인
```

#### 테스트 3-2: 최소 발언 수 변경
```
1. Settings 탭 → "최소 발언 수" 를 10으로 변경
2. 테스트 회의 진행 (15회 발언)
3. Stop Recording 클릭
4. 학습이 시작되는지 확인 (15 >= 10)
```

## 🐛 예상 문제 및 해결

### 문제 1: "PEFT not available" 에러

**확인:**
```bash
python -c "from train_persona import TRAIN_AVAILABLE; print(TRAIN_AVAILABLE)"
```

**해결:**
```bash
pip install peft==0.17.1 transformers==4.56.2 accelerate==1.10.1
```

### 문제 2: "CUDA out of memory" 에러

**확인:**
```python
import torch
print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

**해결:**
- Batch size 줄이기 (core/persona_training_worker.py에서 수정)
- 다른 GPU 프로그램 종료
- CPU로 학습 (느리지만 가능)

### 문제 3: UI가 멈춤

**원인:** QThread가 제대로 작동하지 않음

**확인:**
```python
# meeting_console.py에서
print(f"Worker running: {worker.isRunning()}")
```

**해결:**
- 앱 재시작
- 학습 중 다른 작업 최소화

## 📊 성능 벤치마크

### 테스트 환경
```
- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3070 (8GB)
- RAM: 32GB
- OS: Windows 11
```

### 예상 학습 시간

| 발언 수 | 데이터셋 크기 | 학습 시간 (GPU) | 학습 시간 (CPU) |
|--------|-------------|----------------|----------------|
| 20개   | ~100개 페어  | ~5분           | ~30분          |
| 50개   | ~250개 페어  | ~10분          | ~60분          |
| 100개  | ~500개 페어  | ~15분          | ~120분         |

### 메모리 사용량

| 구성 요소 | 메모리 사용 |
|---------|-----------|
| 베이스 모델 (4-bit) | ~2GB VRAM |
| LoRA 어댑터 | ~500MB VRAM |
| 학습 배치 | ~1GB VRAM |
| **총계** | **~3.5GB VRAM** |

## ✅ 성공 기준

### 필수 조건
- [ ] 회의 종료 시 자동으로 학습 시작
- [ ] 진행 상황이 UI에 표시됨
- [ ] 학습 완료 후 알림창 표시
- [ ] 어댑터 파일이 정상 생성됨
- [ ] 페르소나에 어댑터 경로 저장됨

### 선택 조건
- [ ] 학습 중 UI가 멈추지 않음
- [ ] 에러 발생 시 적절한 메시지 표시
- [ ] 설정 변경이 즉시 반영됨
- [ ] 여러 화자 동시 학습 가능 (순차적)

## 🔍 디버깅 팁

### 로그 확인
```
Status 탭에서 다음 메시지 확인:

정상 흐름:
1. "회의 종료: N명 참여자 기록 업데이트"
2. "⏭ speaker_X: 발언 수 부족 (...) - 학습 건너뜀" (발언 부족 시)
   또는
   "🧠 speaker_X QLoRA 학습 시작 (발언: N개)"
3. "📊 speaker_X 데이터셋 생성 중..."
4. "🧠 speaker_X 말투 학습 중..."
5. "✅ speaker_X 학습 완료!"
6. "   어댑터 저장 위치: adapters/..."
7. "   페르소나에 어댑터 경로 저장됨"

에러 발생 시:
- "❌ 학습 실패: ..."
- "⚠ RAG Store 없음 - 학습 불가"
- "⚠ 페르소나 업데이트 실패: ..."
```

### 파일 시스템 확인
```bash
# 데이터셋 생성 확인
cat data/persona_datasets/speaker_*_dataset_*.jsonl | head -5

# 어댑터 메타데이터 확인
cat adapters/speaker_*/final/metadata.json

# 페르소나 저장 확인
cat data/digital_personas/speaker_*.json | grep qlora_adapter_path
```

### 코드 레벨 디버깅
```python
# meeting_console.py에서 추가
def _trigger_auto_training(self, speaker_ids: List[str]):
    print(f"[DEBUG] Auto training triggered for: {speaker_ids}")
    print(f"[DEBUG] Auto training enabled: {self.auto_training_enabled}")
    print(f"[DEBUG] Min utterances: {self.min_utterances_for_training}")
    # ... 기존 코드
```

## 📝 테스트 결과 보고

### 테스트 완료 체크리스트

```markdown
## QLoRA 자동 학습 테스트 결과

**날짜:** YYYY-MM-DD
**테스터:** 이름
**버전:** v1.0.0

### 시나리오 1: 회의 녹음 후 자동 학습
- [ ] 설정 확인: 통과
- [ ] 테스트 회의 진행: 통과
- [ ] 학습 진행 확인: 통과
- [ ] 결과 확인: 통과
- [ ] 페르소나 사용: 통과

**결과:** ✅ 성공 / ❌ 실패
**비고:**

### 시나리오 2: 오프라인 파일 처리
- [ ] 파일 준비: 통과
- [ ] 파일 업로드: 통과
- [ ] 자동 학습: 통과

**결과:** ✅ 성공 / ❌ 실패
**비고:**

### 시나리오 3: 수동 설정 변경
- [ ] 자동 학습 끄기: 통과
- [ ] 최소 발언 수 변경: 통과

**결과:** ✅ 성공 / ❌ 실패
**비고:**

### 발견된 문제
1.
2.

### 개선 제안
1.
2.
```

## 🎓 추가 학습 자료

- **QLoRA 논문:** https://arxiv.org/abs/2305.14314
- **PEFT 문서:** https://huggingface.co/docs/peft
- **Transformers 문서:** https://huggingface.co/docs/transformers
