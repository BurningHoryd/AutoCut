import os
import re
import numpy as np
import librosa
from scipy.signal import correlate
import moviepy.editor as mp
from datetime import datetime
import cv2
import pytesseract
from fuzzywuzzy import fuzz
import easyocr
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

# Tesseract 실행 파일의 경로를 로컬 경로로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 비디오 파일이 있는 폴더 설정
video_folder = 'input'  # 비디오 파일들이 있는 폴더 경로
start_sound_path = 'start_sound.wav'  # 시작 사운드 파일
middle_sound_path = 'middle_sound.wav'  # 중간 사운드 파일
end_sound_paths = ['end_sound_1.wav', 'end_sound_2.wav']  # 끝 사운드 파일 리스트

# 낮은 샘플링 레이트 설정
low_sr = 22050  # 예: 22.05 kHz

# 파일 이름에서 유효하지 않은 문자 제거
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename).replace('.', '_')

# 비디오 파일 목록을 가져옴
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.mov', '.MOV'))]

for video_file in video_files:
    video_path = os.path.abspath(os.path.join(video_folder, video_file))  # 비디오 파일 경로
    print(f"Processing file: {video_file}")

    # MP4 파일 경로 설정
    mp4_file_path = os.path.splitext(video_path)[0] + '.mp4'

    # MP4 파일이 이미 존재하는지 확인
    if os.path.exists(mp4_file_path):
        print(f"MP4 file already exists: {mp4_file_path}. Skipping conversion.")
        video_path = mp4_file_path  # 변환된 MP4 파일 경로로 변경
    else:
        # MOV 파일을 MP4로 변환
        if video_file.endswith('.mov') or video_file.endswith('.MOV'):
            print(f"Converting to MP4: {mp4_file_path}")

            # FFmpeg 명령어로 변환 (멀티스레딩 활성화)
            os.system(f'ffmpeg -i "{video_path}" -c:v libx264 -preset ultrafast -crf 23 -threads 0 "{mp4_file_path}"')
            
            video_path = mp4_file_path  # 변환된 MP4 파일 경로로 변경


    # 비디오의 오디오를 추출합니다.
    print("Starting to extract audio...")
    video = mp.VideoFileClip(video_path)
    # 강제로 덮어쓰기 위해 -y 옵션 추가
    output_audio_path = os.path.join(os.getcwd(), 'extracted_audio.wav')
    os.system(f'ffmpeg -y -i "{video_path}" -map 0:a:0 "{output_audio_path}"')
    print("Audio extraction complete.")

    # 중복된 시간을 제거하는 함수 (1초 이내의 중복을 하나 병합)
    def remove_close_times(times, tolerance=1.0):
        if not times:
            return []
        sorted_times = sorted(times)
        merged_times = []
        current_group = [sorted_times[0]]

        for time in sorted_times[1:]:
            if time - current_group[-1] <= tolerance:
                current_group.append(time)
            else:
                # 그룹의 평균을 대표값으로 사용
                merged_times.append(round(np.mean(current_group), 2))
                current_group = [time]
        # 마지막 그룹 추가
        merged_times.append(round(np.mean(current_group), 2))
        return merged_times

    # 비디오 오디오 로드
    print("Loading audio for correlation...")
    y, sr = librosa.load('extracted_audio.wav', sr=low_sr)  # 낮은 샘플링 레이트로 로드
    print("Audio loading complete.")

    # 시작 사운드 로드
    print("Loading start sound...")
    start_y, _ = librosa.load(start_sound_path, sr=low_sr)
    print("Start sound loading complete.")

    # 중간 사운드 로드
    print("Loading middle sound...")
    middle_y, _ = librosa.load(middle_sound_path, sr=low_sr)
    print("Middle sound loading complete.")

    # 컷을 저장할 리스트
    segments = []
    print("Starting to find sounds...")

    # 시 사운드 찾기
    print("Finding start sound...")
    start_correlation = correlate(y, start_y, mode='valid')
    start_threshold = 0.75  # 시작 사운드 유사도 기준
    start_frames = np.where(start_correlation >= start_threshold * np.max(start_correlation))[0]

    # 중간 사운드 찾기
    print("Finding middle sound...")
    middle_correlation = correlate(y, middle_y, mode='valid')
    middle_threshold = 0.3  # 중간 사운드 유사도 기준을 0.3으로 설정
    middle_frames = np.where(middle_correlation >= middle_threshold * np.max(middle_correlation))[0]

    # 끝 사운드 찾기
    end_frames_all = []
    for end_sound_path in end_sound_paths:
        print(f"Finding end sound for {end_sound_path}...")
        end_y, _ = librosa.load(end_sound_path, sr=low_sr)  # 끝 사운드 로드
        end_correlation = correlate(y, end_y, mode='valid')
        end_threshold = 0.75  # 유사도 기준
        end_frames = np.where(end_correlation >= end_threshold * np.max(end_correlation))[0]
        end_frames_all.append(end_frames)  # 모든 끝 사운드 프레임을 저장

    # 시작, 중간, 끝 프레임을 초 단위로 변환
    print("Converting start frames to times...")
    start_times = [round(frame / low_sr, 2) for frame in start_frames]
    start_times = remove_close_times(start_times, tolerance=1.0)
    print(f"Start times after removing close detections: {start_times}")

    print("Converting middle frames to times...")
    middle_times = [round(frame / low_sr, 2) for frame in middle_frames]
    middle_times = remove_close_times(middle_times, tolerance=1.0)
    print(f"Middle times after removing close detections: {middle_times}")

    print("Converting end frames to times...")
    end_times = []
    for end_frames in end_frames_all:
        end_times.extend([round(frame / low_sr, 2) for frame in end_frames])
    end_times = remove_close_times(end_times, tolerance=1.0)
    print(f"End times after removing close detections: {end_times}")

    # 시작, 중간, 끝을 순서대로 매칭
    print("Matching start, middle, and end times...")
    clip_segments = []
    
    # 정렬된 시간들
    start_times_sorted = sorted(start_times)
    middle_times_sorted = sorted(middle_times)
    end_times_sorted = sorted(end_times)

    i = j = k = 0  # start, middle, end 인덱스
    while i < len(start_times_sorted) and j < len(middle_times_sorted) and k < len(end_times_sorted):
        start = start_times_sorted[i]
        middle = middle_times_sorted[j]
        end = end_times_sorted[k]

        if start < middle < end:  # 올바른 순서인 경우
            clip_segments.append((start, middle, end))
            i += 1
            j += 1
            k += 1
        elif middle <= start:  # 중간이 시작보다 앞에 있는 경우
            j += 1
        elif end <= middle:  # 끝이 중간보다 앞에 있는 경우
            k += 1
        else:  # 시작이 중간보다 뒤에 있는 경우
            i += 1

    print("Clip segments (start, middle, and end times):")
    for i, (start, middle, end) in enumerate(clip_segments):
        print(f"Clip {i + 1}: Start at {start:.2f} seconds, Middle at {middle:.2f} seconds, End at {end:.2f} seconds")


    # 클립 저장할 디렉토리 설정
    output_directory = 'clips/'  # 클립을 저장할 디렉토리

    # 저장할 디렉토리가 없으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 비디오 파일 이름에서 날짜 추출
    video_filename = os.path.basename(video_path)  # 경로에서 파일 이름만 추출
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', video_filename)

    if date_match:
        shooting_date = date_match.group(1)
    else:
        shooting_date = datetime.now().strftime("%Y-%m-%d")

    # 원본 비디오 로드
    video = mp.VideoFileClip(video_path)

    def extract_victory_or_defeat(clip):
        # 클립 끝나기 1초 전 프레임에서 결과 분석
        snapshot = clip.get_frame(clip.duration - 1)
        
        # 이미지 전처리
        gray = cv2.cvtColor(snapshot, cv2.COLOR_RGB2GRAY)  # RGB를 Grayscale로 변환
        inverted = cv2.bitwise_not(gray)  # 색깔 반전

        # 왼쪽 위 영역 지정 (예: x=0, y=0, width=200, height=50)
        roi = inverted[0:170, 0:450]  # y, x 순서로 지정
        
        # OCR을 사용하여 텍스트 추출
        result_text = pytesseract.image_to_string(roi, config='--psm 6').strip()  # OCR 결과
        print(f"OCR Result: {result_text}")  # OCR 결과 확인

        result_text = re.sub(r'[^가-힣A-Za-z0-9]', '', result_text)  # 한글, 문, 숫자 제외한 모든 문자 제거
        print(f"Cleaned OCR Result: {result_text}")  # 정리된 OCR 결과 확인

        # 유사도 기준 설정
        victory_keywords = ["VICTORY", "승리"]
        defeat_keywords = ["DEFEAT", "패배"]

        # 유사도 점수 계산
        victory_similarity = max(fuzz.ratio(result_text, keyword) for keyword in victory_keywords)
        defeat_similarity = max(fuzz.ratio(result_text, keyword) for keyword in defeat_keywords)

        # 승리/패배 판별
        if victory_similarity >= defeat_similarity:
            return "Victory"
        else:
            return "Defeat"
        


    def extract_game_mode(clip, output_dir="output_images"):
        # 클립 시작 후 2초 시점의 프레임을 추출
        snapshot = clip.get_frame(2)

        # 이미지 전처리
        gray = cv2.cvtColor(snapshot, cv2.COLOR_RGB2GRAY)
        inverted = cv2.bitwise_not(gray)  # 흰색 텍스트를 검은색으로 반전
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 저장할 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 이미지 저장
        cv2.imwrite(os.path.join(output_dir, "thresh_image.png"), thresh)

        # 게임 모드 영역 추출 (두 줄의 텍스트를 모두 포함하는 영역을 선택)
        game_mode_roi = thresh[350:660, 600:1350]

        # 게임 모드 영역도 저장
        cv2.imwrite(os.path.join(output_dir, "game_mode_roi.png"), game_mode_roi)

        # OCR 적용 (EasyOCR)
        reader = easyocr.Reader(['ko', 'en'])
        results = reader.readtext(game_mode_roi)

        # 결과 리스트가 비어 있으면 "Unknown" 반환
        if not results:
            return "Unknown"

        # OCR 결과 병합 (한  병합)
        full_text = " ".join([res[1] for res in results]).strip()
        print(f"OCR Result: {full_text}")

        # 유효한 텍스트만 필터링 (길이 기준, 숫자/특수문자 제거)
        cleaned_text = re.sub(r'[^A-Za-z가-힣\s]', '', full_text).strip()

        print(f"Cleaned OCR Result: {cleaned_text}")  # 정리된 OCR 결과 확인

        # 필터링된 텍스트가 없으면 "Unknown" 반환
        if len(cleaned_text) < 3:
            return "Unknown"

        # 유사도 비교
        game_modes = ["Splat Zones", "Tower Control", "Rainmaker", "Clam Blitz", "Turf war"]
        best_mode = "Unknown"
        best_score = 0

        similarities = [fuzz.ratio(cleaned_text, mode) for mode in game_modes]
        max_similarity = max(similarities)
        if max_similarity > best_score:
            best_score = max_similarity
            best_mode = game_modes[np.argmax(similarities)]

        # 결과 출력
        print(f"Detected Game Mode: {best_mode} with similarity score: {best_score}")

        return best_mode

    # 아이디 순서 찾는 함수
    def find_player_id(clip, player_id="HIGOO", id_positions=[(325, 390, 20, 465), (475, 540, 20, 465), (625, 685, 20, 465), (775, 840, 20, 465)]):
        # 클립 시작 후 2초 시점의 프레임을 출
        snapshot = clip.get_frame(2)

        # 이미지 전처리 (그레이스케일로 변환)
        gray = cv2.cvtColor(snapshot, cv2.COLOR_RGB2GRAY)

        # OCR 리더 초기화
        reader = easyocr.Reader(['ko', 'en'])

        best_match_index = None  # 가장 유사한 인덱스를 저장할 변수
        best_similarity_score = 0  # 가장 높은 유사도 점수

        for i, (y1, y2, x1, x2) in enumerate(id_positions):
            # 지정된 위치에서 ROI 추출
            roi = gray[y1:y2, x1:x2]

            # OCR을 사용하여 텍스트 추출
            results = reader.readtext(roi)

            # OCR 결과에서 내 아이디 찾기
            for (bbox, text, prob) in results:
                cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text).strip()  # 텍스트 정리
                
                # 유사도 점수 계산
                similarity_score = fuzz.ratio(cleaned_text, player_id)

                # 유사도 점수가 더 높으면 업데이트
                if similarity_score > best_similarity_score:
                    best_similarity_score = similarity_score
                    best_match_index = i + 1  # 1부터 시작하는 인덱스

        if best_match_index is not None:
            print(f"Player ID '{player_id}' found at position {best_match_index} with similarity score: {best_similarity_score}.")
            return best_match_index  # 가장 유사한 위치 반환

        print(f"Player ID '{player_id}' not found in any of the positions.")
        return None  # 아이디를 찾지 못한 경우


    def overlay_images(base_image, overlay_image):
        # overlay_image가 RGBA일 경우
        if overlay_image.shape[2] == 4:
            # 알파 채널이 있는 경우
            alpha_channel = overlay_image[:, :, 3] / 255.0  # 알파 값을 0~1로 정규화
            overlay_rgb = overlay_image[:, :, :3]  # RGB만 출
            
            # base_image overlay_image를 겹치기
            for c in range(3):  # R, G, B 채널에 대해
                base_image[:, :, c] = alpha_channel * overlay_rgb[:, :, c] + (1 - alpha_channel) * base_image[:, :, c]
        else:
            # 알파 채널이 없는 경우
            base_image = cv2.add(base_image, overlay_image)  # 그냥 덧셈

        return base_image

    def mse_lab(imageA, imageB):
        # LAB 색상 공간으로 변환
        labA = cv2.cvtColor(imageA, cv2.COLOR_BGR2Lab)
        labB = cv2.cvtColor(imageB, cv2.COLOR_BGR2Lab)

        err = np.sum((labA.astype("float") - labB.astype("float")) ** 2)
        err /= float(labA.shape[0] * labA.shape[1])
        return err

    def load_reference_image(image_path):
        # 참조 이미지를 로드하고 윤곽선 찾기
        reference_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, reference_thresh = cv2.threshold(reference_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(reference_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0] if contours else None  # 첫 번째 윤곽선 반환


    def mse(imageA, imageB):
        # MSE 계산
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def extract_contours(image):
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 이진화 (Adaptive Thresholding 사용)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 윤곽선 추출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0] if contours else None  # 첫 번째 윤곽선 반환

    def shape_similarity(roi, reference_contour):
        # ROI 이미지에서 윤곽선 찾기
        roi_contour = extract_contours(roi)
        
        if roi_contour is None or reference_contour is None:
            return 1.0  # 윤곽선이 없을 경우 1 반환 (완전히 다름으로 간주)

        # 형 유사도 계산
        match_score = cv2.matchShapes(reference_contour, roi_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        
        # 예를 들어, match_score가 클수록 형태가 다르다고 가정
        # match_score가 작을수록 유사하므로, 정규화
        max_score = 100  # 적절한 최대값 설정 (사전에 실험하여 결정)
        normalized_score = min(match_score / max_score, 1.0)  # 1.0보다 큰 경우를 처리

        return normalized_score

    def compute_ssim(imageA, imageB):
        # 이미지가 비어있지 않은지 확인
        if imageA is None or imageB is None:
            print("One of the images is None.")
            return 0  # 0 반환 (완전 다름으로 간주)
        
        # 크기 확인
        if imageA.shape != imageB.shape:
            print("Images do not have the same size.")
            return 0  # 0 반환 (완전히 다름으로 간주)
        
        # 이미지가 이미 그레이스케일인지 확인
        if len(imageA.shape) == 2:  # 그레이스케일
            grayA = imageA
        else:  # 컬러 이미지인 경우
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)

        if len(imageB.shape) == 2:  # 그레이스케일
            grayB = imageB
        else:  # 컬러 이미지인 경우
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # SSIM 계산
        score, _ = ssim(grayA, grayB, full=True)
        return max(score, 0)  # 음수를 0으로 변경


    def get_roi(image, x, y, width, height):
        return image[y:y+height, x:x+width]


    def find_weapon_icon(clip, start_time, player_position, weapon_folder='weapons', output_folder='output_images'):
        # 플레이어의 x축 좌표 설정
        x_positions = {
            1: (523, 614),
            2: (611, 702),
            3: (700, 791),
            4: (788, 879)
        }

        # 플레이어의 위치에 맞는 x축 범위 가져오기
        if player_position not in x_positions:
            print("Invalid player position.")
            return "Unknown"

        x1, x2 = x_positions[player_position]
        y1, y2 = 23, 114  # 고정된 y축 좌표

        # 데이터베스에 있는 모든 무기 아이콘을 로드
        weapon_images = {}
        for weapon_name in os.listdir(weapon_folder):
            weapon_path = os.path.join(weapon_folder, weapon_name)
            weapon_image = cv2.imread(weapon_path, cv2.IMREAD_UNCHANGED)  # RGBA로 읽기
            if weapon_image is not None:
                weapon_images[weapon_name] = cv2.resize(weapon_image, (91, 91), interpolation=cv2.INTER_LINEAR)

        # 참력 폴더가 존재하지 않으면 생성
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        best_match_name = None
        best_similarity_score = float('inf')
        best_overlay_image = None

        # ROI 이미지를 저장할 리스트
        roi_images = []

        # 시작 시점부터 5초간 프레임 추출
        for seconds in range(6):  # 0, 1, 2, 3, 4, 5초
            current_time = start_time + seconds
            print(f"Extracting frame at {current_time} seconds from clip start")
            
            # 현재 시점에서 프레임을 추출
            snapshot = clip.get_frame(current_time)

            # 색상 공간 변환: RGB에서 BGR로 변환
            snapshot = cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR)

            weapon_roi = snapshot[y1:y2, x1:x2]  # 무기 아이콘 영역 추출

            # ROI를 저장
            roi_images.append(weapon_roi)

        # 형태 유사도 필터링을 위한 리스트
        filtered_roi_images = roi_images  # errcheck를 통한 필터링을 제거하였으므로 모든 ROI를 사용

        # 각 무기 아이콘과 비교 (filtered_roi_images 용)
        for weapon_name, weapon_image in weapon_images.items():
            for roi in filtered_roi_images:  # 필터링된 이미지를 사용
                # 클립 ROI 위에 데이터베이스 이미지를 겹치기
                overlay_image = overlay_images(roi.copy(), weapon_image)

                # 유사도 계산 (MSE_LAB 사용)
                similarity_score = mse_lab(roi, overlay_image)  # MSE로 대체

                # 유사도가 기준을 과할 때만 최고 유사도 업데이트
                if similarity_score < best_similarity_score:  # MSE가 낮을수록 유사함
                    best_similarity_score = similarity_score
                    best_match_name = weapon_name
                    best_overlay_image = overlay_image  # 겹친 이미지 저장

        # 최고 유사도를 가진 무기 이름 반환
        if best_match_name:
            best_match_name_without_extension = os.path.splitext(best_match_name)[0]
            print(f"Detected weapon: {best_match_name_without_extension} with lowest MSE: {best_similarity_score:.2f}")

            # 최고 유사도 이미지를 저장 (덮어쓰지 않도록)
            if best_overlay_image is not None:
                overlay_filename = os.path.join(output_folder, f'best_overlay_image_{best_match_name_without_extension}.png')
                cv2.imwrite(overlay_filename, best_overlay_image)  # 최고 유사도 겹친 이미지 저장
                print(f"Saved best overlay image: {overlay_filename}")

            return best_match_name_without_extension  # 무기 이름 반환

        print("No valid weapon icon found.")
        return "Unknown"  # 무기 아이콘을 찾지 못한 경우


    def find_result_frame(clip, end_time, max_search_time=20, output_folder='result_images'):
        """끝 사운드 이후 HIGOO 텍스트가 있는 프레임 찾기"""
        reader = easyocr.Reader(['en'])  # 영어 텍스트 인식용
        
        # 결과 이미지 저장할 폴더 생성
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 0.5초 간격으로 프레임 검사
        for t in range(int(end_time), int(end_time + max_search_time), 1):
            try:
                frame = clip.get_frame(t)
                # BGR로 변환
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # HIGOO가 있을 것으로 예상되는 영역 추출
                # 좌표는 실제 게임 화면에 맞게 조정 필요
                roi = frame_bgr[270:310, 1067:1152]  # y1:y2, x1:x2
                
                # OCR 수행
                results = reader.readtext(roi)
                
                # HIGOO 텍스트 찾기
                for (bbox, text, prob) in results:
                    if 'HIGOO' in text.upper():
                        print(f"Found HIGOO at {t:.2f} seconds")
                        # HIGOO를 찾은 전체 프레임 저장
                        frame_filename = os.path.join(output_folder, f'result_frame_{t}.png')
                        cv2.imwrite(frame_filename, frame_bgr)
                        print(f"Saved result frame at {t} seconds: {frame_filename}")
                        return t, frame_bgr
                        
            except Exception as e:
                print(f"Error processing frame at {t} seconds: {e}")
                continue
                
        return None, None

    def extract_kill_death(frame, frame_time):
        """결과 화면에서 Kill/Death 정보를 추출"""
        try:
            # 각 자리수 ROI 추출
            kill_tens = frame[293:316, 1519:1534]    # 킬 십의 자리
            kill_ones = frame[293:316, 1534:1548]    # 킬 일의 자리
            death_tens = frame[293:316, 1597:1612]   # 데스 십의 자리
            death_ones = frame[293:316, 1612:1626]   # 데스 일의 자리
            
            def match_number(roi):
                best_match = None  # 가장 유사한 숫자
                min_diff = float('inf')
                
                # kdcheck 폴더의 모든 템플릿과 비교
                for template_file in os.listdir('kdcheck'):
                    if not template_file.endswith('.png'):
                        continue
                    
                    template_path = os.path.join('kdcheck', template_file)
                    template = cv2.imread(template_path)
                    
                    if template is None or template.shape != roi.shape:
                        continue
                    
                    # 이미지 차이 계산 (MSE)
                    diff = np.mean((roi.astype("float") - template.astype("float")) ** 2)
                    
                    # 가장 작은 차이를 가진 숫자 선택
                    if diff < min_diff:
                        min_diff = diff
                        # 파일 이름에서 숫자 추출 (예: "1.png" -> 1)
                        best_match = int(os.path.splitext(template_file)[0])
                
                return best_match  # 유사한 숫자가 없으면 None 반환
            
            # 각 자리수 인식
            k_tens = match_number(kill_tens)
            k_ones = match_number(kill_ones)
            d_tens = match_number(death_tens)
            d_ones = match_number(death_ones)
            
            # 결과 조합
            kills = k_tens * 10 + k_ones
            deaths = d_tens * 10 + d_ones
            
            print(f"Detected K/D: {kills}/{deaths}")
            return kills, deaths
            
        except Exception as e:
            print(f"Error extracting K/D: {e}")
            return None, None




    # 클립 저장 부분
    for i, (start, middle, end) in enumerate(clip_segments): 
        # 클립의 끝 시간을 3초 연장
        end_extended = end + 3.0
        
        # 전체 클립 생성
        clip = video.subclip(start, end_extended)
        
        # 레이어 아이디 확인
        player_position = find_player_id(clip)
        
        # 무기 인식 (middle 시점 + 4초부터 5초)
        weapon_name = find_weapon_icon(clip, middle - start + 4.0, player_position, weapon_folder='weapons')

        # 승리/패배 정보 추출
        result = extract_victory_or_defeat(clip)

        # 게임 모드 및 맵 정보 추출
        game_mode = extract_game_mode(clip)

        # 무기 정보 처리 (특수문자 제거 및 공백 처리)
        if weapon_name is not None:
            weapon_info = re.sub(r'[.]', '', weapon_name)  # . 제거
            weapon_info = weapon_info.replace('_', ' ')  # _를 공백으로 변경
        else:
            weapon_info = "Unknown"  # 무기를 찾지 못한 경우

        # 과 프레임 찾기
        result_time, result_frame = find_result_frame(clip, end - start)
        
        if result_frame is not None:
            # K/D 추출
            kills, deaths = extract_kill_death(result_frame, result_time)
            kd_info = f"K{kills}D{deaths}" if kills is not None and deaths is not None else "UnknownKD"
        else:
            print("Result screen not found")
            kd_info = "UnknownKD"

        # 파일명 생성 (형식: "YYYYMMDD_승리or패배_게임모드_무기_KD_클립번호")
        base_filename = f"{shooting_date.replace('-', '')}_{result}_{sanitize_filename(game_mode)}_{sanitize_filename(weapon_info)}_{kd_info}_{i + 1}"
        output_path = os.path.join(output_directory, f"{base_filename}.mp4")
        
        # 파일이 이미 존재하는 경우 새로운 번호 추가
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_directory, f"{base_filename}_{counter}.mp4")
            counter += 1
            
        try:
            # FFmpeg를 직접 사용하여 클립 추출 (모든 오디오 트랙 유지)
            ffmpeg_command = (
                f'ffmpeg -hide_banner -nostdin -stats -i "{video_path}" '
                f'-ss {start} -t {end_extended - start} '
                f'-map 0:v:0 -map 0:a -c:v libx264 -c:a aac '
                f'-preset ultrafast -y "{output_path}"'
            )
            print(f"Executing FFmpeg command: {ffmpeg_command}")
            os.system(ffmpeg_command)
            print(f"Saved clip with all audio tracks: {output_path}")
            
        except Exception as e:
            print(f"Error during clip extraction: {e}")
            # 실패 시 MoviePy로 시도
            try:
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    preset='ultrafast'
                )
                print(f"Saved clip using MoviePy: {output_path}")
            except Exception as e:
                print(f"MoviePy fallback also failed: {e}")

         # 클립 저장 대신 더미 파일 경로 출력
        #print(f"Dummy clip path: {output_path}")
        #print(f"Start: {start}, End: {end_extended}")
        #print(f"Player Position: {player_position}")
        #print(f"Weapon: {weapon_info}")
        #print(f"Result: {result}")
        #print(f"Game Mode: {game_mode}")
        #print("-------------------")

        # 클립 객체 정리
        clip.close()

    # 클립 저장 완료
    print(f"All clips processed for {video_file}.")
