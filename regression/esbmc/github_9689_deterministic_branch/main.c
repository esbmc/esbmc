// #9689: branches fixed by the program text get no branching waypoint.
int sum(int n)
{
  if (n < 1)
    return 0;
  return n + sum(n - 1);
}

int main(void)
{
  int r = sum(5);
  __ESBMC_assert(r != 15, "reached");
  return 0;
}
