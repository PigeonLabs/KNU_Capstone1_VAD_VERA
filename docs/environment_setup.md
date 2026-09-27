# 환경 구성과 재검산

1단계는 Python 3.13.13, PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128 기반 환경과 VERA 전용 overlay 환경에서 실행했다. 정확한 목록은 requirements.lock.txt와 experiments/vera_ipad/requirements-vera.lock.txt, 당시 전체 관측 목록은 environment.txt를 참고한다.

## 공개 결과만 CPU로 검산

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python numpy scikit-learn
.venv/bin/python scripts/verify_publication.py
```

원본 이미지/가중치가 없어도 저장된 점수, 지표, 비교 파일, 해시를 검증한다. 위 최소 검산 환경은 원 추론 환경과 다르다.

## 원 추론 환경 구성

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/bin/python scripts/bootstrap_vera.py --install
cache/vera/venv/bin/python scripts/download_vera_models.py
```

IPAD R01–R04의 원본 프레임을 IPAD_dataset/{R01..R04}/{training,testing}/frames/{video}/와 test_label/{video}.npy 구조로 별도 확보해야 한다. 실제 촬영 FPS가 검증되지 않아 30 FPS를 가정했다. GPU 모델/정밀도를 임의로 바꾸면 새로운 비교 실험으로 기록한다.

1단계 원 실행 명령은 아래와 같았다. 공개 저장소에는 완료 구간이 이미 포함되어 있어 동일 명령은 기존 결과를 재사용한다. 완전히 독립적인 재실행 또는 후속 실험은 원형 증거를 덮어쓰지 않는 별도 결과 경로와 실행기를 만든 뒤 수행한다.

```bash
PYTHONPATH=. HF_HOME="$PWD/cache/vera/hf" cache/vera/venv/bin/python scripts/run_vera.py --phase all
cache/vera/venv/bin/python scripts/analyze_vera.py
cache/vera/venv/bin/python -m pytest -q tests/test_vera.py tests/test_disk_guard.py
```

새 실행에는 scripts/logged_command.py를 사용한다. 디스크 여유 10 GiB 이하에서는 중단하고 자동 재개하지 않는다. 프로토콜과 캐시를 보존하며 원형 자료의 재생성/덮어쓰기를 피한다.
