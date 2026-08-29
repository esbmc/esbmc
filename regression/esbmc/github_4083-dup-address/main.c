// Taking one label's address twice must yield one value: the number a label
// gets is its first position in the function's address-taken set, not its
// per-occurrence position. Pins that `&&L` is a stable value rather than a
// fresh one per mention (issue #4083).
int main(void)
{
  void *a = &&L;
  void *b = &&L;
  __ESBMC_assert(a == b, "one label, one address");
  goto *a;
L:
  return 0;
}
