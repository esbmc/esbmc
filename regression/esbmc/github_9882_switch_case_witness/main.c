/* #9882: a switch gets one branching waypoint naming the case taken. */
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
  __ESBMC_assert(y != 7, "case 7 taken");
  return 0;
}
