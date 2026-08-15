int main()
{
  int x = nondet_int();
  _Bool b;

  if (x > 5)
  {
    b = nondet_bool();
    __ESBMC_assert(x > 0, "an unrelated interval fact survives an unmodelled write");
  }
  return 0;
}
