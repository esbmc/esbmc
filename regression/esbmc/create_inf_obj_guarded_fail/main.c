#include <assert.h>

struct __ESBMC_PyObj
{
  const void *value;
  unsigned long type_id;
  unsigned long size;
};

struct __ESBMC_PyObj *__ESBMC_create_inf_obj(void);
int nondet_int(void);

struct L
{
  unsigned long size;
  struct __ESBMC_PyObj *items;
};

int main()
{
  int n = nondet_int();
  if (n > 0 && n < 4)
  {
    // Allocated under a path guard: items must stay the array's address
    // while size changes, so each access below resolves to one element.
    struct L l;
    l.items = __ESBMC_create_inf_obj();
    l.size = 0;
    for (int i = 0; i < n; i++)
    {
      l.items[l.size].size = i;
      l.size++;
    }
    assert(l.items[l.size - 1].size == n);
  }
  return 0;
}
