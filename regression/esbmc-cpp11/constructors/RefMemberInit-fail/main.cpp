#include <cassert>

struct a {
  a() : c(b) { b = 42; }
  int &c;
  int b;
} d;
int main() {
  assert(d.c == 0);
}
