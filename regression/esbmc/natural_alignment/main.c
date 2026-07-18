#include <assert.h>
#include <stdint.h>

// Objects with no alignment attribute are still naturally aligned:
// their address is a multiple of alignof(T) (C11 6.2.8).

int g_int;
double g_double;
int *g_ptr;
int g_arr[8];

struct plain
{
  int a;
  int b;
};
struct plain g_struct;

int main()
{
  int l_int;
  double l_double;
  int l_arr[8];
  struct plain l_struct;

  assert((uintptr_t)&g_int % _Alignof(int) == 0);
  assert((uintptr_t)&g_double % _Alignof(double) == 0);
  assert((uintptr_t)&g_ptr % _Alignof(int *) == 0);
  assert((uintptr_t)&g_arr % _Alignof(int) == 0);
  assert((uintptr_t)&g_struct % _Alignof(struct plain) == 0);

  assert((uintptr_t)&l_int % _Alignof(int) == 0);
  assert((uintptr_t)&l_double % _Alignof(double) == 0);
  assert((uintptr_t)&l_arr % _Alignof(int) == 0);
  assert((uintptr_t)&l_struct % _Alignof(struct plain) == 0);

  return 0;
}
