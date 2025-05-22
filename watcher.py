import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RestartOnChangeHandler(FileSystemEventHandler):
    def __init__(self, script):
        self.script = script
        self.process = subprocess.Popen([sys.executable, script])

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print("Code changed, restarting app...")
            self.process.terminate()
            self.process.wait()
            self.process = subprocess.Popen([sys.executable, self.script])

if __name__ == "__main__":
    script_to_run = "testinhouse.py"  # yahan aapka main PyQt5 script ka naam likho
    event_handler = RestartOnChangeHandler(script_to_run)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        event_handler.process.terminate()
    observer.join()
