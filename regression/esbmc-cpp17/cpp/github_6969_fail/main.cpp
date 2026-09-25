extern "C" void __ESBMC_assert(bool, const char *);

/* The negative twin: a bodyless closure returns nondet, which satisfies any
 * assertion the solver likes. This one is false in every instantiation, so a
 * regression back to nondet returns would let it pass. */
template <typename T> static int f(int v)
{
  auto g = [](int x) { return x > 0 ? 10 : 20; };
  return g(v);
}

int main()
{
  __ESBMC_assert(f<long>(1) == 20, "long instantiation is 10, not 20");
  return 0;
}
