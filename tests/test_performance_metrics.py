from src.metrics.collector import MetricsCollector


def test_current_frame_quality_reuses_existing_stb_formulas_without_ending_target():
    collector = MetricsCollector(run_id="run-quality")
    collector.start_target("performance_flow", 0, 0)
    collector.add_frame(face_detected=True, gaze_valid=True, timestamp=1.0)
    collector.add_frame(face_detected=True, gaze_valid=False, timestamp=1.1)
    collector.add_frame(face_detected=False, gaze_valid=False, timestamp=1.2)

    quality = collector.get_current_frame_quality()

    assert quality == {
        "stb01_fps": 10.0,
        "stb02_landmark_rate": 0.6667,
        "stb03_face_fail_rate": 0.3333,
        "stb04_dropout_rate": 0.6667,
    }
    assert collector.is_measuring() is True
