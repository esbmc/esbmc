// The loop writes y only through p, and p never moves: the havoc covers *p,
// so y is state.
int main() {
  int y = 0;
  int *p = &y;
  unsigned char x = 0;
  for (;;) {
    __ESBMC_assert(y < 4, "y bounded");
    *p = x;
    x = (x + 1) & 3;
  }
}
