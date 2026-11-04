/* #9882: a GNU case range is a case, named by its low bound, not default. */
int nondet_int(void);

int main()
{
  int y = 0;
  switch (nondet_int())
  {
  case 1:
    y = 1;
    break;
  case 5 ... 7:
    y = 5;
    break;
  default:
    y = 3;
  }
  __ESBMC_assert(y != 5, "range taken");
  return 0;
}
