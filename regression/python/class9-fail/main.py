class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self.balance = balance

account = BankAccount("Alice", 100.0)
assert account.owner == "Alic"