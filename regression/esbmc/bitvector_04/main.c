#include <assert.h>

int main() {
  _ExtInt(10) x = nondet_float();
  _ExtInt(10) y = nondet_int();
  _ExtInt(10) z = x + y;
  assert(z == x + y);
}
