// g's loop gets the k-induction havoc followed by an unconditional assume
// of its condition, false for g(0): extraction must refuse, or that assume
// cuts off the path to the assertion.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
unsigned g(unsigned n) { unsigned s = 0; for (unsigned i = 0; i < n; i++) s++; return s; }
int main() {
  unsigned char x = 0;
  while (x < 3) x++;
  unsigned r = g(0);
  __VERIFIER_assert(r != 0);
  return 0;
}
