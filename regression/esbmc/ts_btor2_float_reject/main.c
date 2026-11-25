// BTOR2 has no floating-point sort, so --ts-check rejects the export.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  float f = 0.0f;
  for (;;) {
    __VERIFIER_assert(f >= 0.0f);
    f = f + 1.0f;
  }
}
