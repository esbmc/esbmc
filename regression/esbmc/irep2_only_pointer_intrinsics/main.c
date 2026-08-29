/* do_special_functions lowers three intrinsics by their __ESBMC_ name rather
   than a __builtin_ prefix. Left as calls the symbols are bodyless and
   goto_check rejects the program outright -- offsetof expands to the first of
   them, so any use of <stddef.h>'s macro was fatal under this flag. */
#include <assert.h>
#include <stddef.h>

struct s
{
  int x;
  int y;
};

int a, b;

int main(void)
{
  assert(offsetof(struct s, y) == sizeof(int));
  assert(__ESBMC_same_object(&a, &a));
  assert(!__ESBMC_same_object(&a, &b));
  /* Distinct objects, so this discriminates the lowering: every intrinsic
     here yields 0 as an offset, and only the object id separates them. */
  assert(__ESBMC_POINTER_OBJECT(&a) != __ESBMC_POINTER_OBJECT(&b));
  return 0;
}
