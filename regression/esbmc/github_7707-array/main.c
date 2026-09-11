#include <stdint.h>

/* Residual of #7707: build_reference_to() routes an array-typed object to
 * bounds_check() rather than check_data_obj_access(), and construct_from_array()
 * recurses into a structure subtype before its own alignment check, so no
 * alignment claim is generated for an element of an array of packed structs. */

struct __attribute__((packed)) S
{
  char a;
  char pad[7];
  uint64_t b;
};

struct S arr[4];

int main(void)
{
  uint64_t *p = (uint64_t *)&arr[1].b;
  uint64_t z = *p;
  (void)z;
  return 0;
}
