int nondet_int(void);

int main(void)
{
  int a[4];
  int i = nondet_int();
  int j = nondet_int();
  if (i >= 0 && i < 4)
    a[i] = 1;
  if (j != 0)
    return 10 / j;
  return 0;
}
