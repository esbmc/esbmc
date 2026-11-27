extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  while (x < 10)
  {
    if (__VERIFIER_nondet_int())
      break;
    x = x + 1;
  }
  __VERIFIER_assert(x <= 10);
  return 0;
}
