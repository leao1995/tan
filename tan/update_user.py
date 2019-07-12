import os
import sys
import StringIO
import smtplib

old_stdout = sys.stdout # Memorize the default stdout stream
buffer = StringIO.StringIO()

def email_recipient(message):
    if not message:
        message = "This message was empty"
    
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    
    # start TLS for security 
    s.starttls() 
    
    # Authentication 
    s.login("****", "****") 
    
    # sending the mail 
    s.sendmail("experiment_results@lupalab.com", "sakbar@ncsu.edu", message) 
    
    # terminating the session 
    s.quit() 

def add_buffer_to_io():
    sys.stdout = buffer

def remove_buffer_from_io():
    sys.stdout = old_stdout
    buffer.close()

def update_user_by_email():
    experiment_data = buffer.getvalue() # Return a str containing the entire contents of the buffer.
    print(experiment_data) # Why not print it?
    email_recipient(experiment_data)

def run_experiment(exper):
    add_buffer_to_io()
    try:
        exper()
    except Exception as e:
        print(e)
    update_user_by_email()
    remove_buffer_from_io()