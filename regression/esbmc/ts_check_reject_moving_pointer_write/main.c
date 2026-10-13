// p moves inside the loop, so the havoc covers p itself, and a pointer cannot
// be TS state.
int main() {
  int a[2] = {0, 0};
  int *p = a;
  unsigned char x = 0;
  for (;;) {
    __ESBMC_assert(a[0] < 4, "a bounded");
    *p = x;
    p = (p == a) ? a + 1 : a;
    x = (x + 1) & 3;
  }
}
