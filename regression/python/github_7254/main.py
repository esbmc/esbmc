def withdraw(balance, amount):
    assert amount <= balance


def process(initial_balance, withdrawals):
    withdraw(initial_balance, withdrawals[0])


process(100, [20])
