<h1> Creamo STT </h1>
<hr/>
Extract timestamps and text per speaker from the video to extract features
<br/><br/>
<h1> 🔈 Project Introduction </h1>
<hr/>
Train the voices of the teacher and the child to separate speakers, then extract the conversation content of each speaker per timestamp.<br/>
Then, analyze these utterances to extract features.
<br/><br/>
<h1>:calendar: When? </h1>
<hr/>
230703-230829 , KIST_Creamo
<br/><br/>
how to run?
1. Put mp4 file into video folder (by user)
2. Run CreeTT.ipynb
   Then the below will run automatically.

   RUN   
	-> convert video to audio   
	-> ectract text   
	-> save text in timeline folder   
	-> check the spelling   
	-> Split voice file according to timestamp (speaker_test folder)   
	-> teacher / student voice classification (make teacher & student folder in speaker_test folder -> audio file)   
	-> teacher / student text classification (make teacher & student in timeline folder -> text file)   
	-> Organize folders   
   ->extract features (TBD / ex.Extracting highly repeated words & Similarity check for each word)    
     
speaker_learn folder is in my google drive (too big to put in my github)
