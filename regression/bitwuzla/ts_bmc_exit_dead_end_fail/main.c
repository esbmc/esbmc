// Exiting paths are dead ends, but the path that keeps counting reaches the
// violation at step 3.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern void exit(int);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  for (;;) {
    unsigned char in = __VERIFIER_nondet_uchar();
    if (in == 0)
      exit(0);
    __VERIFIER_assert(x != 3);
    x = x + 1;
  }
}
