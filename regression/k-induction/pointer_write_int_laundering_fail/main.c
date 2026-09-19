// q may hold b's address, laundered through an integer.
extern int __VERIFIER_nondet_int(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int a[2], b[2];

int main()
{
  long zero = 0;
  long addr = (long)&b[0] + zero;
  int *q = &a[0];
  int i = 0;
  if (__VERIFIER_nondet_int())
    q = (int *)addr;
  for (;;)
  {
    q[1] = q[1] + 1;
    i++;
    __VERIFIER_assert(b[1] < 10);
  }
}
