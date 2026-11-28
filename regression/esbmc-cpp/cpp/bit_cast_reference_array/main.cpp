#include <cassert>
int one[1] = {42};
int (&a1())[1] { return one; }
int main() {
  int v = __builtin_bit_cast(int, a1());
  assert(v == 42);
  return 0;
}
