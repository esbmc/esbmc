// #7964: p leaves its loop-entry object; the inductive step must not assume it stays.
int a[1];
int b[5];
int main()
{
  int *p = a;
  int s = 0;
  while (1)
  {
    if (__ESBMC_same_object(p, b))
    {
      s += *p; // reads b[5] on the seventh iteration
      p++;
    }
    else
      p = b;
  }
}
