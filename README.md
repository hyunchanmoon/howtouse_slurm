# HPC 클러스터 사용 가이드
> Slurm 기반 GPU 작업 환경 설정 및 실행

---

## 1. Slurm 개요

Slurm은 HPC 클러스터에서 자원을 관리하고 작업을 스케줄링하는 오픈소스 시스템입니다.  
CPU, GPU, 메모리 등을 효율적으로 배분하여 여러 사용자가 클러스터를 공유할 수 있게 합니다.

| 명령어 | 설명 |
|--------|------|
| `srun` | 자원 할당 + 즉시 실행 |
| `salloc` | 자원만 먼저 예약 (대화형) |
| `sbatch` | 스크립트 기반 백그라운드 배치 실행 |
| `sinfo` | 클러스터 파티션 및 노드 상태 확인 |
| `squeue` | 대기 중인 작업 목록 확인 |
| `scancel` | 작업 취소 |

---

## 2. 클러스터 상태 확인

작업 제출 전 `sinfo`로 파티션 및 노드 상태를 확인합니다.

```bash
$ sinfo
```

| 상태 | 의미 |
|------|------|
| `idle` | 여유 있음 - 즉시 사용 가능 |
| `mix` | 일부 사용 중 - GPU 남아있을 수 있음 |
| `alloc` | 전부 사용 중 - 대기 필요 |
| `plnd` | 예약됨 |

> **idle 또는 mix 상태 파티션을 우선 선택하세요.**

---

## 3. 자원 할당 (salloc)

`salloc`으로 자원을 확보한 뒤, `srun`으로 대화형 bash 세션에 접속합니다.

```bash
# 1단계 - 자원 할당
$ salloc -p cas_v100nv_4 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --mem=70G --comment python

# 2단계 - 노드 접속
$ srun --pty /bin/bash -l
```

| 옵션 | 설명 |
|------|------|
| `-p` | 파티션 이름 지정 |
| `--ntasks` | 프로세스 수 (GPU 작업은 보통 1) |
| `--cpus-per-task` | 태스크당 CPU 코어 수 |
| `--gres=gpu:N` | GPU 개수 |
| `--mem` | 메모리 (예: 70G) |
| `--comment` | 작업 유형 명시 (필수) |

> **메모리 계산:** `cpus-per-task × 9178MB` 이하로 설정  
> 예) 8 CPU → 최대 약 73GB

---

## 4. 모듈 로드

노드 접속 후 필요한 CUDA 모듈을 로드합니다.

```bash
$ module purge
$ module load cuda/12.1
```

> `cuda/12.1`은 deprecated로 자동으로 `cuda/12.4.1`로 리다이렉트됩니다. 정상 동작합니다.

---

## 5. 가상환경 설정 (uv)

`uv`는 pip보다 10~100배 빠른 Python 패키지 관리 도구입니다.

### uv 설치

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ source ~/.bashrc
```

### 가상환경 생성 및 활성화

```bash
$ uv venv torch21 --python 3.11
$ source torch21/bin/activate
```

> **주의:** torch 2.1.0은 Python 3.11까지만 지원 (3.12 미지원)

### 패키지 설치

```bash
# torch 계열은 whl 주소 별도 지정 필요
$ uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 나머지는 requirements.txt로 설치
$ uv pip install -r requirements.txt
```

### torch 버전 호환표

| torch | torchvision | torchaudio |
|-------|-------------|------------|
| 2.1.0 | 0.16.0 | 2.1.0 |
| 2.2.0 | 0.17.0 | 2.2.0 |

---

## 6. 배치 작업 실행 (sbatch)

반복 실행이나 장시간 작업은 sbatch 스크립트로 제출합니다.

### run.sh 예시

```bash
#!/bin/bash
#SBATCH -p cas_v100nv_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=70G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ria-pruning
#SBATCH --comment python           # 필수 옵션
#SBATCH --output=ria_pruning_%j.out
#SBATCH --error=ria_pruning_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com

module purge
module load cuda/12.1

source torch21/bin/activate

echo "=== Environment Check ==="
nvidia-smi
python --version
echo "GPU count: $SLURM_GPUS_ON_NODE"

cd $SLURM_SUBMIT_DIR

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Starting Job ==="
uv run python main.py \
    --model /scratch/x3433a02/open_weight_model/decapoda-research-llama-7B-hf \
    --prune_method ria \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save_model pruned/llama_7b/2-4/ria/
echo "=== Job Complete ==="
```

### 제출 방법

```bash
# Windows에서 작성한 경우 DOS 줄바꿈 제거 필수
$ sed -i 's/\r//' run.sh
$ sbatch run.sh
```

> **주의:** `--comment` 옵션 필수. `python`, `pytorch`, `tensorflow` 등 중 선택

---

## 7. 전체 워크플로우 요약

```
1. sinfo                          # 클러스터 상태 확인 → idle/mix 파티션 선택
2. salloc -p <partition> ...      # 자원 할당
3. srun --pty /bin/bash -l        # 노드 접속
4. module load cuda/12.1          # CUDA 로드
5. source torch21/bin/activate    # 가상환경 활성화
6. python main.py ... 또는        # 직접 실행
   sbatch run.sh                  # 배치 제출
```
