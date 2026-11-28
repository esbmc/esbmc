/* Issue #11131: --sarif-output wrote no file under --multi-property. */
int nondet_int(void);
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
