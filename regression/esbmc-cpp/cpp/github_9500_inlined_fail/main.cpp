#include <cassert>
int nondet_int();
void f(bool r, int *p) noexcept;

void h(bool r, int *p) {
  f(r, p);
}

void f(bool r, int *p) noexcept {
  int l = nondet_int();
  if (r) {
    *p = 20;
    __ESBMC_assume(l == 30);
  } else {
    h(true, &l);
    assert(l != 20);
  }
}

int main() {
  f(false, nullptr);
  return 0;
}
