int main()
{
  int *p, *q;
  int x=10, y=11;

  p = &x;
  q = &y;

  assert(__ESBMC_same_object(p, p));

  return 0;
}
