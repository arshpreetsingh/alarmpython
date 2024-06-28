from alarm import Alarm
from break_message import BreakMessage
from progress_message import ProgressMessage
import schedule
import time
import requests


"""def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {filename}")


def job_alarm():
    alarm.start_alarm()"""


def job_break_message():
    break_message.show_message1()


def progress_break_message():
    progress_message.show_progress()


if __name__ == "__main__":
    # Get user input for alarm sound URL, break message, and progress total steps
    alarm_sound_url = input("Enter the URL to the alarm sound file: ")
    break_message_text = input("Enter the break message: ")
    progress_total_steps = (input("Enter the progress Message: "))

    # Download the alarm sound file
    """alarm_sound_file = "alarm_sound.mp3"
    download_file(alarm_sound_url, alarm_sound_file)

    alarm = Alarm(alarm_sound_file)"""
    break_message = BreakMessage(break_message_text)
    progress_message = ProgressMessage(progress_total_steps)

    # Schedule the alarm
    #alarm_time = input("Enter the alarm time (in HH:MM format): ")
    #schedule.every(1).minutes.do(job_alarm)

    # Schedule the break message
    schedule.every(1).minutes.do(job_break_message)

    #progress message schedule
    schedule.every(2).minutes.do(progress_break_message)

    while True:
        schedule.run_pending()
        time.sleep(1)
