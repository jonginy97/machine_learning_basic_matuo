# 📘 Lecture 03: PyTorch 기초 요약

## 1. PyTorch 개요
- Facebook에서 개발한 **동적 계산 그래프 기반** 딥러닝 프레임워크
- TensorFlow, Theano와 달리 실행 시점에 계산 그래프를 구성
- 자연어 처리, 비정형 구조 데이터 처리에 강함

## 2. Tensor 기본

### 2.1 텐서 생성
- `torch.tensor`, `torch.ones`, `torch.zeros`, `torch.full`, `torch.eye` 등으로 생성
- NumPy 배열로부터도 생성 가능

### 2.2 자료형 지정
- `dtype=torch.float`, `torch.int`, `torch.long` 등

### 2.3 크기 확인
- `.size()`, `.shape`  
- 특정 차원만 얻기: `.size(dim)`

### 2.4 텐서 변형
- `.reshape()`, `.view()`, `.unsqueeze()`, `.squeeze()`
- `.transpose()`, `.permute()`, `.split()`, `.chunk()`
- `.cat()`, `.stack()`

### 2.5 연산
- 산술 연산: `+`, `-`, `*`, `/`, `log`, `exp`, `sqrt`
- 집계 연산: `sum`, `mean`, `var`, `std`, `max`
- 노름: `torch.norm`
- 행렬 곱: `dot`, `matmul`, `bmm`, `einsum`
- 조건: `torch.where`, `torch.clamp`, `torch.eq`, `torch.gt` 등

### 2.6 변환
- `.numpy()`, `.tolist()`, `.item()`

### 2.7 디바이스 설정
- `.to(device)`, `.cuda()`, `.cpu()`
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## 3. Autograd (자동 미분)
- `.requires_grad = True` 설정 시, 자동으로 미분 계산
- `.backward()`로 역전파 수행
- `.grad`로 기울기 조회
- `.detach()`, `torch.no_grad()`로 계산 차단 가능

## 4. 신경망 모델 구성 (`nn.Module`)
- `nn.Module`을 상속하여 레이어 구성
- `__init__`과 `forward()`를 오버라이드하여 정의
- `nn.Sequential`로 연속된 네트워크 정의 가능

## 5. 최적화 (`torch.optim`)
- `optim.SGD`, `Adam` 등 다양한 옵티마이저 제공
- `.zero_grad()` → `.backward()` → `.step()` 순서로 학습 진행

## 6. 모델 학습 흐름
1. 데이터 로딩 (ex. `torchvision.datasets`)
2. 모델 정의 (`MLP`, `Sequential` 등)
3. 손실 함수 정의 (예: 크로스 엔트로피)
4. 옵티마이저 설정
5. 학습 루프
   - 순전파
   - 손실 계산
   - 역전파
   - 파라미터 업데이트
6. 검증 루프 (모델 평가용)

## 7. 모델 저장 및 불러오기
- 저장: `torch.save(model.state_dict(), "model.pth")`
- 불러오기:
  ```python
  model.load_state_dict(torch.load("model.pth"))
  ```

## 8. `torchvision` & `DataLoader`
- 데이터셋: `torchvision.datasets.MNIST`, `CIFAR10`, 등
- 전처리: `transforms.Compose([...])`
- 미니배치 처리: `torch.utils.data.DataLoader(...)`