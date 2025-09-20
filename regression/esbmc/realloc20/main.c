#include <assert.h>
#include <stdlib.h>

unsigned nondet_uint();

void test_vector_growth()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n <= 20); // symbolic but bounded

  int *vec = malloc(sizeof(int));
  unsigned size = 1;

  for (unsigned i = 0; i < n; ++i)
  {
    if (i == size)
    {
      size *= 2;
      vec = realloc(vec, size * sizeof(int));
      assert(vec != NULL);
    }
    vec[i] = i;
  }

  for (unsigned i = 0; i < n; ++i)
    assert(vec[i] == i);

  free(vec);
}

