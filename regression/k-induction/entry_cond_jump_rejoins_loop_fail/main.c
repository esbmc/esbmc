// The c != 0 path jumps past the back edge and back into the body at mid, so
// that jump does not leave the loop and c == 0 must not be assumed.
int nondet_int();
int main()
{
  int y = 0;
  int c = nondet_int();
  while (1)
  {
    __ESBMC_assert(!(c && y == 15), "y != 15");
    y++;
    if (c)
    {
      c = c;
      goto far;
    }
  mid:
    if (y > 1000000)
      break;
  }
  return 0;
far:
  goto mid;
}
