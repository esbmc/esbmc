/* #9882: a default waypoint rules out case 1, so the witness is refuted. */
int nondet_int(void);

int main()
{
  int y = 0;
  switch (nondet_int())
  {
  case 1:
    y = 1;
    break;
  default:
    y = 3;
  }
  __ESBMC_assert(y != 1, "case 1 taken");
  return 0;
}
