#include <esbmc.h>

int main() {
  int x;
  if (x > 5) {
    ESBMC_cover(x == 10);
  }
  return 0;
}