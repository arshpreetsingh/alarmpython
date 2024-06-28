from pygame import mixer


class Alarm:
    def __init__(self, sound_file):
        self.sound_file = sound_file

    mixer.init()

    def start_alarm(self):
        print(f"Alarm time! Playing Sound:")

        mixer.music.load(self.sound_file)
        mixer.music.play()

