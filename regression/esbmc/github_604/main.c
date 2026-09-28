typedef int func();

/* No definition, and no address-taken function of this signature anywhere in
 * the program: the call through f has no candidate target. */
func *fun();

int main()
{
  func *f = fun();
  int x = f ? f() : 0;
  __ESBMC_assert(!x, "x is 0 unless an unknown callee supplied a value");
  return 0;
}
