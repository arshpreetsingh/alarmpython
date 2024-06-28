from easygui import msgbox


class BreakMessage:
    def __init__(self, message):
        self.message = message

    #def break_time(self):
    def show_message1(self):
        msgbox(self.message, title="Break Message")

