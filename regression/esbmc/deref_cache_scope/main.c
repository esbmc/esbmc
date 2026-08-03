// Same, for liveness rather than validity: the array leaves scope each
// iteration while p keeps naming it, so a memoised resolution of *p would
// hide the dangling access after the loop.
int main(void)
{
  int *p = 0;
  for (int i = 0; i < 2; i++)
  {
    int a[4];
    p = a;
    p[0] = 1;
  }
  p[0] = 2;
  return 0;
}
