// A captureless lambda converts to a function pointer through an implicit
// conversion operator that returns the closure's static invoker. Both are
// implicit members, and both used to be skipped, so the conversion was
// bodyless and yielded an invalid pointer; clang also leaves the invoker's
// forwarding body to CodeGen, which never runs here (issue #4077).
int g = 0;

void run(void (*f)(int))
{
  f(5);
}

int main()
{
  void (*p)() = [] { g = 7; };
  p();
  __ESBMC_assert(g == 7, "the lambda body runs through the pointer");

  int (*q)(int, int) = [](int a, int b) { return a * 10 + b; };
  __ESBMC_assert(q(3, 4) == 34, "arguments and return value are forwarded");

  run([](int v) { g = v * 2; });
  __ESBMC_assert(g == 10, "converted at a call site too");

  auto lam = [](int a) { return a + 1; };
  int (*r)(int) = lam;
  __ESBMC_assert(r(1) == lam(1), "pointer and closure calls agree");
  return 0;
}
