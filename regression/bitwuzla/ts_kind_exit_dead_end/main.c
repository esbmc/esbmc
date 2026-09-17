// A return or exit inside the loop ends the path; the state machine stays in
// {0, 1, 2}, so x != 3 holds.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern void exit(int);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  for (;;) {
    unsigned char in = __VERIFIER_nondet_uchar();
    if (in > 6)
      return 1;
    if (in == 6)
      exit(0);
    __VERIFIER_assert(x != 3);
    x = (x + (in & 1)) % 3;
  }
}
