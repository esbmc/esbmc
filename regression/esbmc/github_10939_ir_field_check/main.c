/* Under --ir --struct-fields-check, a union whose full-size member is a
 * source field is read through that field (#10939). */
#include <assert.h>
struct d { unsigned c; };
union u { unsigned w; };
int main(void)
{
  struct d d = {0x11223344u};
  union u v = *(union u *)&d;
  assert(v.w == 0x11223344u);
  return 0;
}
