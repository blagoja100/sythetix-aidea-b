from datetime import date

class Logger:
    def __init__(self, name="AideaLogger"):
        self.name = name

    def log(self, message):
        today = date.today().isoformat()
        print(f"[{today}] [{self.name}] {message}")