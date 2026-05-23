import time
import ctypes
import subprocess
import sys

def training_is_active():
    try:
        # List running python processes with wmic to see if train.py or evaluate.py is still executing
        cmd = 'wmic process where "name=\'python.exe\' or name=\'py.exe\'" get commandline'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        if 'src/train.py' in output or 'src\\train.py' in output or 'src/evaluate.py' in output or 'src\\evaluate.py' in output:
            return True
    except:
        pass
    return False

def show_popup(message):
    # Creates a standard Windows popup box
    ctypes.windll.user32.MessageBoxW(0, message, "Wafer Map AI Agent", 0x40)

def main():
    # Give it a tiny bit of time to ensure the process actually registered
    time.sleep(30)
    
    # Wait until the target processes end
    while training_is_active():
        time.sleep(30)
        
    show_popup("The Wafer Map Model training pipeline has finished! You can now launch Streamlit.")

if __name__ == '__main__':
    main()
