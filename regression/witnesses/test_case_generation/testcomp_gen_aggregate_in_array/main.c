#include <assert.h>

extern int __VERIFIER_nondet_int(void);

struct pair
{
  int a, b;
};

struct pair arr[3];

int main()
{
  /* The element's struct literal sits inside a WITH on the array:
   * `arr == (... WITH [1:={ .a=nondet1, .b=nondet0 }])`. */
  arr[1].b = __VERIFIER_nondet_int();
  arr[1].a = __VERIFIER_nondet_int();
  if (arr[1].a == 4)
    assert(0);
  return 0;
}
