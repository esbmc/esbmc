// C++23 explicit object parameters ("deducing this"). Issue #4085 reported
// these as unsupported, but the failure was `explicit object parameters are
// incompatible with C++ standards before C++2b` -- clang rejecting the source
// because no C++23 mode was requested. Under --std c++23 each form converts.
struct S
{
  int v;
  int by_val(this S self)
  {
    return self.v + 1;
  }
  int by_ref(this S &self)
  {
    return self.v + 2;
  }
  int by_cref(this const S &self)
  {
    return self.v + 3;
  }
};

int main()
{
  S s{10};
  __ESBMC_assert(s.by_val() == 11, "explicit object parameter by value");
  __ESBMC_assert(s.by_ref() == 12, "explicit object parameter by reference");
  __ESBMC_assert(s.by_cref() == 13, "explicit object parameter by const ref");

  // The recursive-lambda idiom deducing this was introduced for.
  int n = 5;
  auto f = [n](this auto const &self, int k) -> int {
    return k <= 0 ? n : self(k - 1);
  };
  __ESBMC_assert(f(3) == 5, "recursive lambda via deducing this");
  return 0;
}
