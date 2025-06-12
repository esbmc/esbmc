class BankAccount:
    def __init__(self, owner: str):
        self.owner = owner

account = BankAccount("Alice")
assert account.owner == "Alice"

