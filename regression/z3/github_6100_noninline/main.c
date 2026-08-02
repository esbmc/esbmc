int impure(int a)
{
  int s = 0;
  for (int i = 0; i < a; i++)
    s += i;
  return s;
}

int main()
{
  int var;
  __ESBMC_assert(__ESBMC_exists(&var, impure(var) == 0), "e");
}
