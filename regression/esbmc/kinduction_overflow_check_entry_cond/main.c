// The loop's exit test x + 1 > 100 runs only after x = i, but k-induction
// copies it into an ASSUME before the loop, where x is still INT_MAX.
int main(void) {
  int x = 2147483647;
  int i = 0;
  while (i < 10) {
    x = i;
    if (x + 1 > 100)
      break;
    i = i + 1;
  }
  return 0;
}
