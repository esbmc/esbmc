// Outer x IS captured at depth 1; inner block has its own x that must
// NOT be rewritten.  Correct transform:
//   - capture outer x
//   - rewrite the outer `int r = x` to `int r = (*cap_x)`
//   - leave `int x = 1; r += x` untouched inside the inner block.
int main()
{
  int x = 10;
  int inner(void)
  {
    int r = x;
    {
      int x = 1;
      r += x;
    }
    return r;
  }
  __ESBMC_assert(inner() == 11, "outer captured, inner shadow intact");
  return 0;
}
