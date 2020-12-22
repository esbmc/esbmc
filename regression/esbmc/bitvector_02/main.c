#include <assert.h>

int main() {
  _ExtInt(1000) x = nondet_float();
  unsigned _ExtInt(1000) y = nondet_int();
  assert(x == (_ExtInt(1000)) y); 
}
