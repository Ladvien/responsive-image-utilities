import os
from time import sleep

BATTERY_DELAY = 150

NUMBER_OF_SESSIONS = 5
SECONDS_BETWEEN_SESSION_CREATION = 60

count_files_cmd = "watch -n 10 'ls -1 | wc -l'"

while True:
    for i in range(NUMBER_OF_SESSIONS):
        cmd = f"screen -Sdm 'session_num_{i}' bash -c 'cd /home/ladvien/responsive-image-utilities; python pi_downloader.py'"
        os.system(cmd)
        print(f"Opened session number = {i}")
        print("Sleepin before opening next session...")
        sleep(SECONDS_BETWEEN_SESSION_CREATION)
    
    print("Going to sleep while threads work...")
    sleep(BATTERY_DELAY)
    print("Cleaning up old sessions...")
    cmd = "pkill -f 'python pi_downloader'"
    os.system(cmd)
