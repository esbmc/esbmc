int nondet_int(void);

// --multi-property reports both violations, so its SARIF document must carry
// one result for each rather than no document at all.
int main(void)
{
  int a[4];
  int i = nondet_int();
  int j = nondet_int();
  if (i == 7)
    a[i] = 1;
  if (j == 0)
    return 10 / j;
  return 0;
}
