#include <assert.h>

int main(void);

unsigned int global_var3;

int main(void) {
  global_var3 = *main;
  assert(global_var3 == *main);
  return 0;
}

