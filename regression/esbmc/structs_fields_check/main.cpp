#include <assert.h>
struct S {
  int *ref;
  S(int &r) : ref(&r) {}
};
int main() {
  int x = 5;
  S s(x);
  (*s.ref)++;  // Should desugar to *(s.ref) = *(s.ref) + 1
  assert(x == 6);
  return 0;
}
