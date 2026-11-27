// The exit through the loop condition contradicts the entry condition, so
// treating that condition as an invariant would hide this bug.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  while (x < 10)
    x = x + 1;
  __VERIFIER_assert(x != 10);
  return 0;
}
