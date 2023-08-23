# CreamoSTT
텍스트 추출, 화자 분리, 특징 추출

how to run?
1. video 폴더에 mp4 넣기 (사용자가 직접 넣어야 함)
2. CreamoSTT.ipynb실행
	-> 영상을 음성으로 변환
	-> 텍스트 추출
	-> 텍스트를 timeline폴더에 저장
	-> 맞춤법검사
	-> 음성파일을 timestamp에 맞게 분할 (speaker_test폴더)
	-> teacher / student 음성구분 (speaker_test폴더에 teacher / student 폴더 생성_음성)
	-> teacher / student 텍스트 구분(timeline폴더에 teacher / student 생성_ 텍스트)
	-> 폴더정리
   ->#특징추출(미완_주석)

speaker_learn폴더는 google drive에 위치(용량 상 github에 넣지못함)
