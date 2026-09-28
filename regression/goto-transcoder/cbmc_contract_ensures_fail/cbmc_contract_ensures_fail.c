int twice(int x)
  __CPROVER_requires(x > 0 && x < 1000)
  __CPROVER_ensures(__CPROVER_return_value == 3 * x)
{
  return 2 * x;
}

int main()
{
  return twice(5);
}
