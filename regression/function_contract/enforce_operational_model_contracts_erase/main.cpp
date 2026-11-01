#include <vector>
int main() {
  std::vector<int> v;
  v.push_back(1); v.push_back(2); v.push_back(nondet_int()); v.push_back(4);
  unsigned k = nondet_uint(); __ESBMC_assume(k < 4);
  v.erase(v.begin() + k);
  __ESBMC_assert(v.size() == 3, "size");
}
