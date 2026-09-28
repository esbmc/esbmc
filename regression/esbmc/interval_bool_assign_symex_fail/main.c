_Bool b;

int main()
{
  b = nondet_bool();
  __ESBMC_assume(b);
  b = nondet_bool();

  int i = 0;
  do
  {
    i++;
    if (i > 1)
      break;
  } while (b);

  __ESBMC_assert(i == 2, "i is 1 when the second nondet bool is false");
}
