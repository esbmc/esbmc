// The assertion before the loop fails for x == 3; the assumption after it
// must not hide that.
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
extern void __VERIFIER_assume(int);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int x = __VERIFIER_nondet_int();
  __VERIFIER_assert(x != 3);
  __VERIFIER_assume(x != 3);
  int i = 0;
  for (;;)
    i++;
}
