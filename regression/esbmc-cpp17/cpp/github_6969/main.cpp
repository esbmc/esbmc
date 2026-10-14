extern "C" void __ESBMC_assert(bool, const char *);

/* A lambda inside a function template is a distinct closure type in every
 * instantiation. Naming the closure after its source location alone made all
 * three collide, so only the first instantiation got an operator() body and
 * the rest returned nondet. */
template <typename T> static int f(int v)
{
  auto g = [](int x) { return x > 0 ? 10 : 20; };
  return g(v);
}

int main()
{
  __ESBMC_assert(f<int>(1) == 10, "int instantiation");
  __ESBMC_assert(f<long>(1) == 10, "long instantiation");
  __ESBMC_assert(f<char>(-1) == 20, "char instantiation");
  return 0;
}
