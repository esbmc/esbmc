#define CHECK(T, D)                                                            \
  {                                                                            \
    T x = nondet_##T();                                                        \
    T q = x / (T)(D), r = x % (T)(D);                                          \
    __ESBMC_assert((long long)q * (D) + r == x, "quotient and remainder");    \
    __ESBMC_assert(r == 0 || (r < 0) == (x < 0), "remainder sign");           \
    __ESBMC_assert(r > -(long long)(D) && r < (long long)(D), "remainder");   \
  }

typedef long long ll;
int nondet_int(void);
ll nondet_ll(void);

int main(void)
{
  CHECK(int, 2)
  CHECK(int, 8)
  CHECK(int, 1 << 30)
  CHECK(ll, 1024)
  CHECK(ll, 1LL << 62)
  return 0;
}
