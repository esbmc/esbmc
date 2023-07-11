#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

typedef struct
{
  uint8_t field1;
  void *ptr;
} big_struct_t;

void *zero_mem(void *buffer, unsigned int length)
{
  return __ESBMC_memset(buffer, 0, length);
  //return memset(buffer, 0, length);
}

int main()
{
  big_struct_t my_big_struct;
  big_struct_t *big_struct = &my_big_struct;
  //big_struct_t *big_struct = __builtin_alloca(sizeof(big_struct_t));

  zero_mem(big_struct, sizeof(big_struct_t));

  uint8_t *ptr = (uint8_t *)big_struct;
  for(unsigned int i = 0; i < sizeof(big_struct_t); i++)
    assert(*(ptr++) == 0);

  return 0;
}
