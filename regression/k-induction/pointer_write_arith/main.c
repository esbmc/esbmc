// The loop writes a[] through *(p + j); the points-to analysis names a, so
// the inductive step stays enabled and proves the bound on the written cell.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int a[4] = {0, 0, 0, 0};
  int *p = a;
  unsigned k = 0;
  for (;;)
  {
    unsigned j = k % 4;
    *(p + j) = (*(p + j) + 1) % 3;
    k++;
    __VERIFIER_assert(a[j] < 3);
  }
}
