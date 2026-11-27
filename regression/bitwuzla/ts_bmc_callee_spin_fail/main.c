// spin's loop writes nothing, so k-induction leaves it alone; with x havocked
// the step can run it more than once, which extraction must refuse.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
void spin(unsigned char n) { while (n > 3) { if (n == 7) reach_error(); } }
int main() {
  unsigned char x = 0;
  while (x < 3)
  {
    spin(x);
    x = x + 1;
  }
  __VERIFIER_assert(x != 3);
  return 0;
}
