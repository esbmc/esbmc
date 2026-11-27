int nondet_int(void);

int main(void)
{
  int a[4] = {0};
  int i = nondet_int(), j = nondet_int();
  if (i == 5 && j == -1)
    return a[i] + a[j];
  return 0;
}
