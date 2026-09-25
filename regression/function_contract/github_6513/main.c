// A zero-length array member gave Phase 2C an empty witness range, which it
// assumed rather than clamped: `0 <= k && k < 0` is false, so every assertion
// after it in the wrapper was discharged and the contract verified vacuously
// (#6513). The assertion below must be reported.
typedef struct
{
  int len;
  int data[0];
} buf_t;

int g;

void f(buf_t *b)
{
  __ESBMC_requires(b != 0);
  __ESBMC_assigns(g);
  __ESBMC_ensures(1);
  g = 1;
  __ESBMC_assert(0, "must be reported");
}

int main()
{
  return 0;
}
