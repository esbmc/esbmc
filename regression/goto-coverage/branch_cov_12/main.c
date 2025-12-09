#include <assert.h>
int main() {
  int valve = nondet_int();  // environment
  int input = nondet_int();  // user input
  while (input) {
    if (!valve) {
      assert(valve == 0); // "privacy"
      valve = 1;
    }
    else if (valve == 1) {
      assert(valve > 0); // "performance"
       valve = 2;
    }
    else if (valve == 2) {
      assert(valve >= 2); // "vulnerability"
      valve = 3;
    }
  }
  assert(valve == 3);
}