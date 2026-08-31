from src.config import SCREEN_H, SCREEN_W
from src.ui import draw_blink_calibration_screen, draw_performance_dashboard


def test_performance_dashboard_renders_complete_summary_to_one_screen():
    summary = {
        "calibrations": {
            "gaze": {
                "calibration_point_count": 16,
                "calib_reproj_rmse_px": 12.0,
                "iris_std_x_norm_mean": 0.01,
                "iris_std_y_norm_mean": 0.02,
            },
            "blink": {
                "open_ear_median": 0.3,
                "closed_ear_median": 0.1,
                "close_threshold": 0.17,
                "open_threshold": 0.21,
                "total_trials": 5,
                "closed_sample_count": 5,
            },
            "mouth": {
                "mar_baseline": 0.2,
                "open_mar": 0.5,
                "mouth_success_rate": 0.8,
            },
            "completed_count": 3,
            "fallback_count": 0,
        },
        "tests": {
            mode: {
                "target_character": target,
                "selected_character": target,
                "success_rate_percent": 100.0,
                "input_duration_ms": 1000.0,
                "incorrect_attempts": 0,
                "confirmation_attempts": 1,
            }
            for mode, target in (("gaze", "물"), ("blink", "밥"), ("mouth", "집"))
        },
        "runtime_quality": {
            "stb01_fps": 30.0,
            "stb02_landmark_rate": 0.95,
            "stb03_face_fail_rate": 0.05,
            "stb04_dropout_rate": 0.1,
        },
        "recommended_input_mode": "gaze",
    }

    canvas = draw_performance_dashboard(summary)

    assert canvas.shape == (SCREEN_H, SCREEN_W, 3)


def test_blink_calibration_screen_renders_existing_ear_progress():
    canvas = draw_blink_calibration_screen(
        "눈을 편하게 뜨세요.",
        ear=0.3,
        progress=0.5,
        remaining=0.6,
        current_trial=1,
        total_trials=5,
    )

    assert canvas.shape == (SCREEN_H, SCREEN_W, 3)
