int twice(int x)
  __CPROVER_requires(x > 0 && x < 1000)
  __CPROVER_ensures(__CPROVER_return_value == 2 * x)
{
  return 2 * x;
}

int main()
{
  int r = twice(5);
  __CPROVER_assert(r == 10, "caller sees the contract");
  return 0;
}
