#include "extern.h"
int main() {
  __ESBMC_assert(value == 42, "extern should be 42");
  return 0;
}