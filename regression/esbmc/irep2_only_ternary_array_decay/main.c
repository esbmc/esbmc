/* A conditional yielding a pointer from array operands: C11 6.3.2.1p3 converts
   each arm to a pointer to its first element. Coerced as a cast of the array
   object instead, the node reaches the encoder as typecast(array, pointer) and
   aborts it with "Unexpected type in int/ptr typecast". */
#include <assert.h>

char a[4] = {1, 2, 3, 4};
char b[4] = {5, 6, 7, 8};

int main(int argc, char **argv)
{
  char *c = argc == 1 ? a : b;
  assert(c[3] == 4 || c[3] == 8);
  return 0;
}
