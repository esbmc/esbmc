import threading


class Worker:
    @staticmethod
    def run() -> None:
        pass


t = threading.Thread(target=Worker.run)
