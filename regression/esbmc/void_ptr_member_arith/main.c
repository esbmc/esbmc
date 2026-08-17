#include <assert.h>

/* Subtracting from a void* that points at a struct member must move the
 * pointer backwards. ESBMC's value-set treats void as element size one but
 * previously dropped the sign of the subtraction, so `vp - 4` resolved to the
 * same offset as `vp + 4`. */
struct item
{
  long a;
  int data;
  int link;
};

int main(void)
{
  struct item it;
  it.a = 0;
  it.data = 7;
  it.link = 5;

  void *vp = (void *)&it.link;
  int *back = (int *)(vp - 4);
  assert(*back == 7);

  int *fwd = (int *)((void *)&it.data + 4);
  assert(*fwd == 5);
  return 0;
}
