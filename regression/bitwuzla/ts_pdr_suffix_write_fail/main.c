// The suffix writes y, defined before the loop but not state: an exit step
// has no successor, so this is not a write outside the havocked state.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0, y = 0;
  while (x < 10)
    x = x + 1;
  y = x;
  __VERIFIER_assert(y != 10);
  return 0;
}
