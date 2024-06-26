from setmsg import SetMsgValue
if __name__=="__main__":
    msg1 = input("tell me Break message")
    msg2 = input("tell me Progress message")
    value = SetMsgValue(msg1,msg2)
    value.print_number()
    value.print_message()