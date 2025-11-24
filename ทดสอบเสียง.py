from gtts import gTTS
import os

lyrics = """
When you get to the top
You ever been to the top?
Just listen, let me tell ya
Hear what you're missin', shut up and listen

[Pre-Chorus]
In the beginning you'll get crazy
Spending all the money you got
No more women to love you now
You gotta go and leave town

[Chorus]
Back on the rocks
Back on the rocks, baby
You gotta keep your mind together
Back on the rocks
Back on the rocks, baby
You gotta go and live forever
...

[Outro]
Back on the rocks, Back on the rocks
Woah oh, Woah oh oh
Back on the rocks, Back on the rocks
Woah oh, Woah oh oh
"""

tts = gTTS(text=lyrics, lang='en')  # ใช้ 'en' สำหรับภาษาอังกฤษ
tts.save("test.mp3")
os.system("start test.mp3")  # สำหรับ Windows
