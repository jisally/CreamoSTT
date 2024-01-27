from google.colab import drive
drive.mount('/content/drive')
!pip install pydub
!apt install ffmpeg

#mp4(영상)을 mp3(음성)을 변환
from pydub import AudioSegment
import os

def mp4_to_mp3(input_path, output_path, bitrate='128k'):
    # MP4 파일을 AudioSegment로 로드합니다.
    audio = AudioSegment.from_file(input_path, format="mp4")

    # MP3로 저장합니다.
    audio.export(output_path, format="mp3", bitrate=bitrate)

# Input and Output directories
input_folder = '/content/drive/MyDrive/ColabNotebooks/video'
output_folder = '/content/drive/MyDrive/ColabNotebooks/audio'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all .mp4 files in the input_folder
mp4_files = [file for file in os.listdir(input_folder) if file.endswith('.mp4')]

# Convert each .mp4 file to .mp3 and save in the output folder
for mp4_file in mp4_files:
    input_mp4_file = os.path.join(input_folder, mp4_file)
    output_mp3_file = os.path.join(output_folder, f"{mp4_file[:-4]}.mp3")
    mp4_to_mp3(input_mp4_file, output_mp3_file)

#텍스트 추출
!pip install -U stable-ts
!pip install -U git+https://github.com/jianfch/stable-ts.git

import os
import stable_whisper
model = stable_whisper.load_model('large')

# 경로 변경
audio_folder = '/content/drive/MyDrive/ColabNotebooks/audio/'

# audio 폴더에서 .mp3 파일 찾기
mp3_files = [file for file in os.listdir(audio_folder) if os.path.splitext(file)[1] == '.mp3']

# 각 파일에 대해 모델 실행
for filename in mp3_files:
    audio_path = os.path.join(audio_folder, filename)
    result = model.transcribe(audio_path)
    result.to_srt_vtt('audio.vtt', segment_level=True, word_level=False)

import os

# Get a list of all .vtt files in the /content directory
vtt_files = [file for file in os.listdir('/content') if file.endswith('.vtt')]

# Create the 'timeline' folder if it doesn't exist
output_folder = '/content/drive/MyDrive/ColabNotebooks/timeline'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for vtt_file in vtt_files:
    input_vtt_file_path = os.path.join('/content', vtt_file)
    output_file_path = os.path.join(output_folder, f"{mp4_file[:-4]}.txt")

    with open(input_vtt_file_path, 'r') as input_file:
        vtt_content = input_file.read()

    with open(output_file_path, 'w') as output_file:
        output_file.write(vtt_content)

#맞춤법 검사
!git clone https://github.com/ssut/py-hanspell.git
cd /content/py-hanspell
!python /content/py-hanspell/setup.py install
cd /content/drive/MyDrive/ColabNotebooks
!pip install konlpy

import chardet
import textwrap
import os
import sys

# 'py-hanspell' 모듈의 경로 추가
py_hanspell_path = "/content/py-hanspell" 
sys.path.append(py_hanspell_path)
from hanspell import spell_checker

import chardet
import textwrap
import os

# 맞춤법 검사 함수
def check_spell_and_save(input_file_path):
    # 맞춤법 검사 및 수정된 문장 저장할 변수
    checked_sentences = []

    # 파일 열기
    with open(input_file_path, 'rb') as f:
        # 파일 내용 읽기
        rawdata = f.read()

    # 파일 인코딩 감지
    result = chardet.detect(rawdata)
    encoding = result['encoding']

    with open(input_file_path, 'r', encoding=encoding) as input_file:
        original_text = input_file.readlines()

    # 원본 텍스트를 500자씩 나누어 맞춤법 검사 및 수정 수행
    for line in original_text:
        line = line.strip()  # 줄바꿈 제거
        chunks = textwrap.wrap(line, width=500)
        for chunk in chunks:
            spelled_sent = spell_checker.check(chunk)
            checked_sentence = spelled_sent.checked
            checked_sentences.append(checked_sentence)

    # 수정된 문장을 출력 파일로 저장
    with open(input_file_path, 'w', encoding=encoding) as output_file:
        for checked_sentence in checked_sentences:
            output_file.write(checked_sentence + '\n')

if __name__ == "__main__":
    colab_notebooks_folder = "/content/drive/MyDrive/ColabNotebooks"  # 'Colab Notebooks' 폴더의 경로
    time_folder = os.path.join(colab_notebooks_folder, "timeline")  # 'timeline' 폴더의 경로

    # 'timeline' 폴더에 있는 파일들 맞춤법 검사
    for file in os.listdir(time_folder):
        if file.endswith(".txt"):
            file_path = os.path.join(time_folder, file)
            check_spell_and_save(file_path)

#음성파일을 timestamp 에 맞게 분할
import os
import re
from pydub import AudioSegment
from datetime import datetime, timedelta

def str_to_milliseconds(time_str: str):
    dt = datetime.strptime(time_str, "%H:%M:%S.%f")
    milliseconds = int((dt - datetime(1900, 1, 1)).total_seconds() * 1000)
    return milliseconds

def extract_segment_from_audio_file(audio_file: str, start_time: str, end_time: str):
    audio = AudioSegment.from_mp3(audio_file)

    start_time_milliseconds = str_to_milliseconds(start_time)
    end_time_milliseconds = str_to_milliseconds(end_time)

    audio_segment = audio[start_time_milliseconds:end_time_milliseconds]

    return audio_segment

def get_time_ranges_from_file(file_path):
    time_ranges = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = re.findall(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', content)
        for start_time, end_time in matches:
            start_time += '000'
            end_time += '000'
            time_ranges.append((start_time, end_time))

    return time_ranges

# def read_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     return lines

# def write_file(file_path, lines):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for line in lines:
#             f.write(line)

# def is_english_line(line):
#     return bool(re.search('[a-zA-Z]', line))

# 경로 변경
input_audio_folder = "audio"
timeline_folder = 'timeline'

# audio & timeline 폴더에서 파일 찾기
mp3_files = [file for file in os.listdir(input_audio_folder) if os.path.splitext(file)[1] == '.mp3']
txt_files = [file for file in os.listdir(timeline_folder) if os.path.splitext(file)[1] == '.txt']

# 파일 경로 생성
input_audio_file_paths = [os.path.join(input_audio_folder, file) for file in mp3_files]
timeline_file_paths = [os.path.join(timeline_folder, file) for file in txt_files]

output_folder = "speaker_test"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for tl_path, audio_path in zip(timeline_file_paths, input_audio_file_paths):

    time_ranges = get_time_ranges_from_file(tl_path)

    for i, (start_time, end_time) in enumerate(time_ranges):
        audio_segment = extract_segment_from_audio_file(audio_path, start_time, end_time)

        output_file_name = f"output_{i+1:04d}.mp3"
        output_file_path = os.path.join(output_folder, output_file_name)

        audio_segment.export(output_file_path, format="mp3")


# for tl_path, audio_path in zip(timeline_file_paths, input_audio_file_paths):
#     lines = read_file(tl_path)
#     filtered_lines = []
#     skip_next_lines = 0

#     for i, line in enumerate(lines):
#         if skip_next_lines > 0:
#             skip_next_lines -= 1
#             continue
#         elif is_english_line(line) and i > 0:  # i > 0는 첫 번째 문장 예외를 처리합니다.
#             skip_next_lines = 1  # 영어 문장 포함된 다음 타임스탬프 건너뛰기
#         else:
#             filtered_lines.append(line)

#     write_file(tl_path, filtered_lines)

#     time_ranges = get_time_ranges_from_file(tl_path)

#     for i, (start_time, end_time) in enumerate(time_ranges):
#         audio_segment = extract_segment_from_audio_file(audio_path, start_time, end_time)

#         output_file_name = f"output_{i+1:04d}.mp3"
#         output_file_path = os.path.join(output_folder, output_file_name)

#         audio_segment.export(output_file_path, format="mp3")

#teacher/student 구분
import os
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from shutil import move 
from sklearn.metrics import accuracy_score

current_directory = os.getcwd()
filename = "speakerlabel.xlsx"

file_path = os.path.join(current_directory, filename)
data = pd.read_excel(file_path, engine='openpyxl')

def extract_mfcc(file_path, duration=2):
    try:
        y, sr = librosa.load(file_path, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return mfcc.mean(axis=1)
    except Exception as e:
        print(f"Error: {e}")
        return None

features, labels = [], []

for index, row in data.iterrows():

    speaker_learn_folder=os.path.join(current_directory, 'speaker_learn')

    file_path = os.path.join(speaker_learn_folder, row["file_name"] + ".mp3")

    speaker = row["speaker"]
    mfcc = extract_mfcc(file_path)

    if mfcc is not None:
        features.append(mfcc)
        labels.append(speaker)

features = np.array(features)
le = LabelEncoder()
labels = le.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=77, random_state=42)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print('예측 정확도: {:.4f}'.format(accuracy_score(y_test,y_pred)))

#예측 정확도
def predict_speaker(audio_file):
    file_path = audio_file
    mfcc = extract_mfcc(file_path)

    if mfcc is not None:
        feature = np.array([mfcc])
        prediction = clf.predict(feature)
        prediction_proba = clf.predict_proba(feature)[0]

        predicted_label = le.inverse_transform(prediction)[0]
        label_probabilities = dict(zip(le.classes_, prediction_proba))
        return predicted_label, label_probabilities
    else:
        return None, None

# 주어진 speaker_test 폴더 내의 모든 mp3 파일에 대해 처리
source_folder = "speaker_test"


for file_name in os.listdir(source_folder):
    if file_name.endswith(".mp3"):
        file_path = os.path.join(source_folder, file_name)
        predicted_speaker, speaker_probabilities = predict_speaker(file_path)
        print(f"Predicted speaker for {file_name}: {predicted_speaker}")

        #예측 정확도
        for label, probability in speaker_probabilities.items():
            print(f"Probability of {label}: {probability * 100:.2f}%")


        # 예측된 스피커에 따라 폴더 생성 및 파일 이동
        target_folder = os.path.join(source_folder, predicted_speaker)
        os.makedirs(target_folder, exist_ok=True)
        target_file_path = os.path.join(target_folder, file_name)
        move(file_path, target_file_path)
        print(f"Moved {file_name} to {target_folder}")


import os

speaker_test_folder = "speaker_test"
timeline_folder = "timeline"

if not os.path.exists(timeline_folder):
    os.makedirs(timeline_folder)

teacher_folder = os.path.join(speaker_test_folder, "teacher")
student_folder = os.path.join(speaker_test_folder, "student")

if not os.path.exists(teacher_folder):
    os.makedirs(teacher_folder)

if not os.path.exists(student_folder):
    os.makedirs(student_folder)

teacher_files = [file for file in os.listdir(teacher_folder) if file.startswith("output_")]
student_files = [file for file in os.listdir(student_folder) if file.startswith("output_")]

# Read specified lines from the source file.
def read_lines_from_file(file_path, lines_to_read):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    return [content[index] for index in lines_to_read]

# Write lines to the specified file.
def write_lines_to_file(file_path, lines):
    with open(file_path, 'a', encoding='utf-8') as f:
        for line in lines:
            f.write(line)

# Copy specified lines from a source file to a destination file.
def copy_lines_to_file(source_file, target_file, lines):
    with open(source_file, 'r', encoding='utf-8') as source:
        content = source.readlines()
        with open(target_file, 'a', encoding='utf-8') as target:
            for line_num in lines:
                target.write(content[line_num -1])

txt_files = [file for file in os.listdir(timeline_folder) if file.endswith(".txt")]
source_file_path = os.path.join(timeline_folder, txt_files[0])

with open(os.path.join(timeline_folder, "teacher.txt"), "w", encoding='utf-8') as f:
    pass

with open(os.path.join(timeline_folder, "student.txt"), "w", encoding='utf-8') as f:
    pass

for teacher_file in teacher_files:
    file_number = teacher_file.split('_')[1].split('.')[0]  # Extract the numeric part from the filename
    source_lines = [(int(file_number) * 2) , (int(file_number) * 2) +1]
    copy_lines_to_file(source_file_path, os.path.join(timeline_folder, "teacher.txt"), source_lines)

for student_file in student_files:
    file_number = student_file.split('_')[1].split('.')[0]  # Extract the numeric part from the filename
    source_lines = [(int(file_number) * 2) , (int(file_number) * 2) + 1]
    copy_lines_to_file(source_file_path, os.path.join(timeline_folder, "student.txt"), source_lines)

#폴더정리
import os
import shutil

new_folder_name = mp4_file[:-4]

# 새 폴더 생성
if not os.path.exists(new_folder_name):
    os.makedirs(new_folder_name)

folders_to_move = ['audio', 'speaker_test', 'timeline']

for folder in folders_to_move:
    current_folder_path = os.path.join(os.getcwd(), folder)
    new_folder_path = os.path.join(os.getcwd(), new_folder_name, folder)

    if os.path.exists(current_folder_path):
        shutil.move(current_folder_path, new_folder_path)
        print(f"'{folder}' 폴더를 '{new_folder_name}' 폴더로 옮겼습니다.")
    else:
        print(f"'{folder}' 폴더가 현재 디렉토리에 없습니다.")

#video 폴더 초기화

video_folder = './video'

#video폴더 내의 모든 파일 삭제
for file in os.listdir(video_folder):
    file_path = os.path.join(video_folder, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

#데이터셋 만들기
import pandas as pd
import numpy as np
import IPython.display as ipd

!pip install webvtt-py

import pandas as pd
import numpy as np
import IPython.display as ipd


import os
import numpy as np
import pandas as pd
import librosa
import webvtt
from IPython.display import display, Audio

# 시간 표시를 초로 변환하는 함수
def simple_hms(s):
    h, m, sec = [float(x) for x in s.split(':')]
    return 3600 * h + 60 * m + sec


# 모든 .vtt 파일 불러오기
vtt_files = [file for file in os.listdir('/content/') if file.endswith('.vtt')]

# 결과를 저장할 DataFrame 생성
df = pd.DataFrame(columns=['start', 'end', 'text'])

# 모든 .vtt 파일에 대해 반복
for vtt_file in vtt_files:
    transcript = webvtt.read('/content/' + vtt_file)

    # 자막 파일의 내용을 DataFrame에 추가
    for x in transcript:
        df = df.append({'start': x.start, 'end': x.end, 'text': x.text}, ignore_index=True)

# 시간 정보를 초로 변환하여 새로운 열로 추가
df['start_s'] = df['start'].apply(simple_hms)
df['end_s'] = df['end'].apply(simple_hms)
df.head()

#유사도 검사
!pip install konlpy

import os
import numpy as np
import pandas as pd
import librosa
import webvtt
from IPython.display import display, Audio
from konlpy.tag import Okt
from gensim.models import Word2Vec
from gensim.matutils import unitvec

# Function to preprocess text and return nouns
def get_nouns(text):
    okt = Okt()
    nouns = okt.nouns(text)
    return nouns

# Function to train Word2Vec model
def train_word2vec_model(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=4, workers=4)
    return model

# Function to calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    return np.dot(unitvec(v1), unitvec(v2))


# 한국어 형태소 분석기 생성
okt = Okt()

# 'text' 열의 각 문장에서 단어 추출하여 리스트에 저장
noun_sentences = [get_nouns(sentence) for sentence in df['text']]

# Train Word2Vec model
word2vec_model = train_word2vec_model(noun_sentences)

# 사용자에게 입력 가능한 단어 리스트 출력
print("사용자가 입력할 수 있는 단어 리스트:")
available_words = list(word2vec_model.wv.index_to_key)
for index, word in enumerate(available_words, 1):
    print(f"{index}. {word}")

# 사용자에게 단어 입력 받기
user_word_index = int(input("단어를 입력하세요 (번호로 입력): "))

if 1 <= user_word_index <= len(available_words):
    user_word = available_words[user_word_index - 1]
    user_word_vector = word2vec_model.wv[user_word]

    # Calculate and display similar words using Word2Vec model and cosine similarity
    similar_words = []
    for word in word2vec_model.wv.index_to_key:
        if word != user_word:
            similarity = cosine_similarity(user_word_vector, word2vec_model.wv[word])
            similar_words.append((word, similarity))

    num = 0
    similar_words.sort(key=lambda x: x[1], reverse=True)
    print(f"'{user_word}'와 가장 유사한 Top 5:")
    for word, similarity_score in similar_words[:5]:
        num += 1
        print(f"{num}. {word} (유사도: {similarity_score:.3f})")
else:
    print("유효하지 않은 번호입니다. 다시 실행해주세요.")

#빈도 시각화(word cloud)
!pip install wordcloud

from collections import Counter
from wordcloud import WordCloud

# 불용어 리스트를 생성합니다.
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다', '또', '거', '게', '이이', '요', '다가', '오', '오오', '잉', '더', '저', '것', '걸', '개', '고', '거', '것']

# 불용어를 제거한 명사들을 저장할 리스트를 생성합니다.
filtered_nouns = [word for sentence in df['text'] for word, pos in okt.pos(sentence) if pos == 'Noun' and word not in stopwords]

# 단어 빈도 계산
word_freq = Counter(filtered_nouns)

# 워드 클라우드를 생성합니다.
font_path = '/content/drive/MyDrive/ColabNotebooks/NanumBarunpenR.ttf'  # 나눔 폰트 경로
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10,
                      font_path=font_path).generate_from_frequencies(word_freq)

# 생성된 워드 클라우드를 matplotlib를 사용하여 표시합니다.
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)

# 워드 클라우드 플롯 표시
plt.title('word cloud')
plt.show()
