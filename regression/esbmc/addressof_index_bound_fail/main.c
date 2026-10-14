// Anti-vacuity twin: folding &a[n] must keep the bound on the object it came
// from. A fold that discarded the symbol would lose this bounds check.
int main(void)
{
  int a[4] = {1, 1, 1, 1};
  int sum = 0;

  for (int *p = a; p != &a[5]; ++p)
    sum += *p;

  return sum;
}
