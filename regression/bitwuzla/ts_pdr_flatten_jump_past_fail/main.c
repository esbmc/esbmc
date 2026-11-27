extern unsigned char __VERIFIER_nondet_uchar(void);
int main() {
  unsigned char x = 0;
  if (__VERIFIER_nondet_uchar())
    goto between;
  while (x < 3)
    x++;
between:
  x = x + 2;
  while (x < 5)
    x++;
  __ESBMC_assert(x < 5, "jump past the first loop");
  return 0;
}
