extern unsigned char __VERIFIER_nondet_uchar(void);
int main() {
  unsigned char x = 0;
  if (__VERIFIER_nondet_uchar())
    goto mid;
  while (x < 5) {
    x++;
  mid:
    x++;
  }
  __ESBMC_assert(x <= 6, "jump in");
  return 0;
}
