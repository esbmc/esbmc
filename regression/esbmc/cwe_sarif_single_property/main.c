int nondet_int(void);

// Without --multi-property the run stops at the first violation, so its
// SARIF document carries exactly one result.
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
