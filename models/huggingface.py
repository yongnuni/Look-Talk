from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(
    repo_id="dorni/SpeakerVid-5M-data-curation-models",
    filename="L2CSNet_gaze360.pkl",
    revision="5c6e04d7fa3321e6228e79162f8ec98466bf308a",
)

shutil.copy(path, "models/L2CSNet_gaze360.pkl")
print("저장 완료: models/L2CSNet_gaze360.pkl")