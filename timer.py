import os

from langchain.chains import LLMChain
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.retrievers import BedrockRetrievalQAChain

# Set your AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_AWS_ACCESS_KEY_ID"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_AWS_SECRET_ACCESS_KEY"
os.environ["AWS_DEFAULT_REGION"] = "YOUR_AWS_REGION"

# Initialize the Bedrock LLM
llm = Bedrock(
    model_id="anthropic.claude-v2",  # Or another Bedrock model
    client=boto3.client('bedrock-runtime')
)

# Initialize the Bedrock retrieval chain
retriever = BedrockRetrievalQAChain.from_llm(
    llm=llm,
    retriever=your_bedrock_retriever  # Replace with your actual Bedrock retriever
)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\nQuestion: {question}"
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Get relevant documents from the knowledge base
docs = retriever.get_relevant_documents(query="your query")

# Extract the relevant content from the documents
retrieved_text = [doc.page_content for doc in docs]

# Run the chain with the retrieved context and your question
response = chain.invoke({"context": retrieved_text, "question": "your question"})

print(response)


















######################################################################
##                     Script for LIfe                              ##
##								    ##
######################################################################
##  This Python script is ment to be made for the betterment of my  ##
##            life with Computers, I love this script!              ##
##              I love all code I have ever written.                ##
##								    ##
##							            ##
######################################################################

# Required Modules
import schedule

from easygui import msgbox
import os
import time
from pygame import mixer

from tornado.ioloop import IOLoop

from apscheduler.scheduler import Scheduler


# Variables

Musicfile = "/home/metal-machine/Dream(Python)/python-times/apple.mp3"


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
    
    schedule.every(15).minutes.do(Play_Music)
    schedule.every(30).minutes.do(Break_Time)
    
   # Play_Music, 'interval', seconds=14404)
   # scheduler.add_job(Play_Music, 'interval', seconds=14408)
   # scheduler.add_job(Play_Music, 'interval', seconds=14412)
   # scheduler.add_job(Play_Music, 'interval', seconds=14419)
   # scheduler.add_job(Play_Music, 'interval', seconds=14424)

   # schd.add_cron_job(Break_Time, second=1800)
    schedule.every(1).hour.do(Progress_Status)
   # scheduler.add_cron_job(Sleeping_Time, )
#    schd.start()

# Execution will block here until Ctrl+C

    #try:
     #   IOLoop.instance().start()
    #except (KeyboardInterrupt, SystemExit):
     #   pass
while True:
    schedule.run_pending()
    time.sleep(1)
