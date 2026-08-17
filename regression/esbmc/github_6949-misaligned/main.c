#include <stdint.h>

#pragma pack(push, 1)
struct pack1_s
{
  char a;
  uint32_t b;
};
#pragma pack(pop)

int main(void)
{
  struct pack1_s obj;
  struct pack1_s *ptr = &obj;
  uint32_t *q = (uint32_t *)&(ptr->b);
  uint32_t z = *q;
  (void)z;
  return 0;
}
