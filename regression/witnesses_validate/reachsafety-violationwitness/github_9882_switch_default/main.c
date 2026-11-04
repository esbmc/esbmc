/* #9882: the default waypoint ESBMC writes for this switch validates. */
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
  __ESBMC_assert(y != 3, "default taken");
  return 0;
}
