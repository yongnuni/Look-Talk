# gaze_backbone.py 의 L2CS 라이브러리 버전
import numpy as np

try:
    import torch
    from l2cs import Pipeline
    _L2CS_AVAILABLE = True
except ImportError:
    _L2CS_AVAILABLE = False

try:
    from src.config import GAZE_MODEL_PATH
except ImportError:
    GAZE_MODEL_PATH = "models/L2CSNet_gaze360.pkl"


class GazeBackbone:

    def __init__(self, model_path=None):
        self.available = False
        self.pipeline = None

        path = model_path or GAZE_MODEL_PATH

        if not _L2CS_AVAILABLE:
            print("[gaze_backbone] l2cs/torch 미설치 → 백본 비활성화 (기하 특징만 사용)")
            return

        import os
        if not os.path.exists(path):
            print(f"[gaze_backbone] 모델 없음: {path} → 백본 비활성화")
            return

        try:
            self.pipeline = Pipeline(
                weights=path,
                arch="ResNet50",
                device=torch.device("cpu"),
            )
            self.available = True
            print(f"[gaze_backbone] L2CS Pipeline 로드 완료: {path}")
        except Exception as e:
            print("[gaze_backbone] 로드 실패:", e)
            self.available = False

    def predict(self, frame, landmarks):
        """
        Returns (gaze_yaw_deg, gaze_pitch_deg) 또는 None.

        landmarks는 인터페이스 호환을 위해 받지만 사용하지 않습니다.
        Pipeline이 내부에서 자체 얼굴 검출을 수행하기 때문입니다.
        (이 중복이 실시간성 부담의 원인이며, ONNX 전환 시 해소됩니다.)
        """
        if not self.available:
            return None

        try:
            results = self.pipeline.step(frame)

            # 얼굴 미검출 시 빈 결과가 올 수 있음
            if results is None:
                return None

            yaw = np.asarray(results.yaw).flatten()
            pitch = np.asarray(results.pitch).flatten()

            if len(yaw) == 0 or len(pitch) == 0:
                return None

            # 여러 얼굴이 잡히면 첫 번째만 사용
            # L2CS 출력은 라디안이므로 degree로 변환 (feature_builder가 degree 전제)
            return float(np.degrees(yaw[0])), float(np.degrees(pitch[0]))

        except Exception as e:
            print("[gaze_backbone] 추론 실패:", e)
            return None