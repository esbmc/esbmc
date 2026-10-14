// Anti-vacuity twin: `*a` writes a[0], not a[1], so the equality is refuted.
int main(void)
{
  int a[3] = {0, 0, 0};

  *a = 7;

  __ESBMC_assert(a[1] == 7, "*a assigns the second element");
  return 0;
}
