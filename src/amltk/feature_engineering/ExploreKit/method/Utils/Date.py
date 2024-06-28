
from datetime import datetime, timedelta

class Date:

    def __init__(self):
        self.time = datetime.now()

    def getSeconds(self) -> int:
        return self.time.second.real

    def getHours(self) -> int:
        return self.time.hour.real

    def getMinutes(self) -> int:
        return self.time.minute.real

    def __sub__(self, other) -> timedelta:
        return self.time - other.time

    def __str__(self):
        return str(self.time)