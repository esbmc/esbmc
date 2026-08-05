#include <assert.h>

/* Negative counterpart: the backwards offset must be modelled precisely
 * enough that reading the wrong member is still detected. */
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
  assert(*back == 5);
  return 0;
}
