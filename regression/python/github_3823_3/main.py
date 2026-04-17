# Global string variable accessed from nested class method
message: str = "hello"

class Printer:
    def get_message(self) -> str:
        return message

p = Printer()
assert p.get_message() == "hello"
