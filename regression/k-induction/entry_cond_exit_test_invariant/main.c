// The exit test reads x and y after the body wrote them, but nothing writes
// them after it, so x >= 0 && y >= 0 holds at every head after the first, and
// at the first because it holds before the loop.
int main()
{
  int x = nondet_int();
  int y = nondet_int();
  if (!(x == y && y >= 0))
    return 0;
  while (x != 0)
  {
    x--;
    y--;
    if (x < 0 || y < 0)
      break;
  }
}
