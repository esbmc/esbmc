#include <list>
template <typename A> static int walk(A &a) {
  int n = 0;
  __ESBMC_unroll(4);
  for (auto i = a.begin(); i != a.end(); ++i) ++n;
  return n;
}
int main() {
  std::list<int> l; unsigned k = nondet_uint(); __ESBMC_assume(k <= 2);
  if (k > 0) l.push_back(1); if (k > 1) l.push_back(2);
  if (nondet_bool() && !l.empty()) l.erase(l.begin());
  __ESBMC_assert(walk(l) <= 2, "w");
}
