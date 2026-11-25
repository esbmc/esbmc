// i++ writes before the break, but not what c < k reads, so the inductive step
// may still assume c < k at the head.
int main()
{
  int k = nondet_int();
  if (k < 0 || k > 100)
    return 0;
  int c = 0, i = 0;
  while (i++ < 1000)
  {
    if (!(c < k))
      break;
    c++;
  }
  __ESBMC_assert(c <= k, "c <= k");
}
