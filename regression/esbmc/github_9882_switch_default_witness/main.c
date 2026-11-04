/* #9882: a switch that matches no case gets one default waypoint. */
int nondet_int(void);

int main()
{
  int y = 0;
  switch (nondet_int())
  {
  case 1:
    y = 1;
    break;
  case 2:
  case 3:
    y = 2;
    break;
  case 7:
    y = 7;
    break;
  default:
    y = 3;
  }
  __ESBMC_assert(y != 3, "default taken");
  return 0;
}
