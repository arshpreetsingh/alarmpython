import time
from easygui import msgbox

title = "GfG - EasyGUI"


class ProgressMessage:

    def __init__(self, total_steps):
        self.total_steps = total_steps

    def show_progress(self):
        msgbox(self.total_steps)
