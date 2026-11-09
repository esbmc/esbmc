// The overflow check on g * 1000000 bounds g, which proves the first loop is
// never entered. Its exit test follows the check on x * 1000, and the bounds
// must not contradict the inductive step's entry assumption there, or the step
// proves the overflow of y below vacuously.
int g;

void set(int *p)
{
  *p = 5;
}

int main()
{
  set(&g);
  int t = g * 1000000;
  int x = g, v, w = 0, y = 2147483645;
  while (x * 1000 > 2147483)
    x = x + 1;
  for (v = 0; v < 4; v++)
    w = w + 1;
  for (v = 0; v < w; v++)
    y = y + 1;
  return t + x + y;
}
