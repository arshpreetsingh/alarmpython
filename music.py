from pygame import mixer
from setmsg import SetMsgValue
Musicfile = "apple.mp3"
class Music:
    def Play_Music(self):
        mixer.init()
        mixer.music.load(Musicfile)
        mixer.music.play()

if __name__=="__main__":
    number = input("tell me number")
    msg = input("tell me message")
    value = SetMsgValue(number, msg)
    value.print_number()
    value.print_message()
