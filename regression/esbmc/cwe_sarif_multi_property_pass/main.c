int nondet_int(void);

// A --multi-property run that proves every claim still writes a well-formed
// SARIF document, with an empty results array.
int main(void)
{
  int a[4];
  int i = nondet_int();
  if (i >= 0 && i < 4)
    a[i] = 1;
  return 0;
}
