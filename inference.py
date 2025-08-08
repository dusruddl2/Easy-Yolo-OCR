import time
import torch
import argparse
from pathlib import Path
from easyocr.easyocr import Reader
from utils.torch_utils import time_sync
from models.experimental import attempt_load
from core.scan import pt_detect


def run_inference(opt):
    img_path = Path(opt.img)
    if not img_path.exists():
        raise FileNotFoundError(f"이미지 경로가 유효하지 않습니다: {img_path}")

    # 디바이스 설정
    device = torch.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    # 모델 로드
    print("[INFO] Detection 모델 로딩 중...")
    detection_model = attempt_load(opt.detection_weights, map_location=device)

    print("[INFO] Recognition 모델 로딩 중...")
    reader = Reader(
        lang_list=['en'],
        model_storage_directory=opt.recog_model_dir,
        user_network_directory=opt.recog_model_dir,
        recog_network=opt.recog_network,
        detector=False,  # YOLO를 detection에 사용하므로 False
        recognizer=True,
        gpu=(opt.gpu >= 0)
    )

    print('[INFO] 모델 로드 완료')

    # 추론 수행
    try:
        start_time = time_sync()
        pred_text, bboxes, confs = pt_detect(
            path = str(img_path),
            device = device,
            models = detection_model,
            ciou = 20,
            reader = reader,
            gray=False,
            byteMode=False,
            img_size=640,
            confidence=0.25,
            iou=0.25
        )
        print(f'🕒 추론 시간: {time_sync() - start_time:.3f}초')
        print(confs)
        print(bboxes)
        print(pred_text)
    except Exception as e:
        print(f"❌ 추론 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=-1, help='GPU 번호 (-1은 CPU)')
    
    opt = parser.parse_args()
    
    opt.img = "./image_ours/LBL124_20250221_20250221_141627_736.png"
    opt.recog_model_dir = "./saved_models/v1.0.0"
    opt.recog_network = "iter_3000"
    opt.detection_weights = opt.recog_model_dir + "/epoch50.pt"
    
    run_inference(opt)