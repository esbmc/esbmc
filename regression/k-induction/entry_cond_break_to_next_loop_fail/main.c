int nondet_int();
int main()
{
  int k = nondet_int(), c = 0, i = 0, j = 0;
  if (k < 0 || k > 100)
    return 0;
  while (i++ < 1000)
  {
    if (!(c < k))
      break;
    c++;
  }
  while (j < 10)
    j++;
  __ESBMC_assert(c < k, "c < k");
}
