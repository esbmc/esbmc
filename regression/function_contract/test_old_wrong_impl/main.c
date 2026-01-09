// Test __ESBMC_old() detects wrong implementation (should FAIL)

int balance = 0;

void withdraw_wrong(int amount)
{
  __ESBMC_requires(amount > 0);
  __ESBMC_requires(balance >= amount);
  __ESBMC_ensures(balance == __ESBMC_old(balance) - amount);

  // BUG: Subtracts double the amount!
  balance -= (amount * 2);
}

int main()
{
  balance = 100;
  withdraw_wrong(30);
  return 0;
}
