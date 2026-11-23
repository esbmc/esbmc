// The loop calls a recursive function, whose writes the check cannot bound, so
// it leaves the loop alone.
int rec(int n)
{
  return n ? rec(n - 1) : 0;
}

int main(void)
{
  int x = 0;
  while (x < 3)
    x += 1 + rec(2);
  return 0;
}
