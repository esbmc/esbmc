#include <stdio.h>

int main(int argc, char* argv[]) {
  __ESBMC_alloc_size[__ESBMC_POINTER_OBJECT(argv)] = (size_t)(argc+1)*sizeof(*argv);

  if (argv[0]) {}
  if (argv[argc]) {}
  int i;
  __ESBMC_assume(0 <= i);
  __ESBMC_assume(i <= argc);
  if (argv[i]) {}

  return 0;
}
