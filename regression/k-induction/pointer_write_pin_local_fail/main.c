// p and q live in main, so their havoced values carry into the next
// iteration: unlike a callee's local, they still need a pin. Unpinned, the
// write through p would be dropped and the inductive step would prove that
// neither a nor b is ever 5.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int a, b;

int main()
{
  int *p = &a, *q = &b, *t;
  unsigned n = 0;
  for (;;)
  {
    a = 0;
    b = 0;
    n++;
    if (n == 6)
      *p = 5;
    __VERIFIER_assert(a != 5 && b != 5);
    t = p;
    p = q;
    q = t;
  }
}
