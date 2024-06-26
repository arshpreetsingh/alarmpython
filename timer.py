######################################################################
##                     Script for LIfe                              ##
##								                                    ##
######################################################################
##  This Python script is ment to be made for the betterment of my  ##
##            life with Computers, I love this script!              ##
##              I love all code I have ever written.                ##
##								                                    ##
##							                                        ##
######################################################################

# Required Modules
"""import schedule

from easygui import msgbox
import os
import time
from pygame import mixer

# Variables

Musicfile = "apple.mp3"


# A function to play music

def Play_Music():
    mixer.init()
    mixer.music.load(Musicfile)
    mixer.music.play()
# time.sleep(10)                  #time.sleep() is used to because play()
# can't hold the program and we can't enjoy musicing. ;)


# A message box to tell you about break time


def Break_Time():


    msgbox(
        'Break-time->Are you standing,,,? -> Are you following your Dream?')


def Progress_Status():

    msgbox(
        "Write your progress in RHINO, If you are feeling happy CONTINUE otherwise see life.jpg on Desktop")


def Sleeping_Time():

    msgbox("Sleeping is as important as Awakening")

if __name__ == '__main__':

   # schd = Scheduler()
   # schd.daemonic = False
   # schd.start()
    
    schedule.every(1).minutes.do(Play_Music)
    schedule.every(2).minutes.do(Break_Time)
    
   # Play_Music, 'interval', seconds=14404)
   # scheduler.add_job(Play_Music, 'interval', seconds=14408)
   # scheduler.add_job(Play_Music, 'interval', seconds=14412)
   # scheduler.add_job(Play_Music, 'interval', seconds=14419)
   # scheduler.add_job(Play_Music, 'interval', seconds=14424)

   # schd.add_cron_job(Break_Time, second=1800)
    schedule.every(1).hours.do(Progress_Status)
   # scheduler.add_cron_job(Sleeping_Time, )
#    schd.start()

# Execution will block here until Ctrl+C

    #try:
     #   IOLoop.instance().start()
    #except (KeyboardInterrupt, SystemExit):
     #   pass
while True:
    schedule.run_pending()
    time.sleep(1)"""
