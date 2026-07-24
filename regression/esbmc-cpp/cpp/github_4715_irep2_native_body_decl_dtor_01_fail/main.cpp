// __ESBMC_assert is a built-in intrinsic; no include needed.

// W1-loc Phase C (esbmc/esbmc#4715): under --irep2-native-body a declaration
// whose type has a destructor (`T t(base);`) or whose initializer needs
// lowering (`T u = T(base + 5);`) is delegated to the legacy convert_decl,
// rather than forcing a whole-function fallback, so the statements around it --
// the plain local and the assignment here -- convert natively while convert_decl
// still emits the in-place construction and schedules each object's scope-exit
// destructor. Both destructors run at the end of the (void) function.
int dtor_count = 0;

struct T
{
  int v;
  T(int a) : v(a) {}
  ~T() { dtor_count++; }
};

int g;

void run(int x)
{
  int base = x;
  T t(base);
  T u = T(base + 5);
  g = t.v + u.v;
}

int main()
{
  run(10);
  __ESBMC_assert(g == 25 && dtor_count == 0, "must fail: both destructors ran");
  return 0;
}
