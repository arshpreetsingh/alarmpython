class SetMsgValue:
    def __init__(self, message1, message2):
        self.message1 = message1
        self.message2 = message2

    def print_number(self):
        print("this is number", self.message1)
        return self.message1
    def print_message(self):
        print("this is message", self.message2)
        return self.message2
