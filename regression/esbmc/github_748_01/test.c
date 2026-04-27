/* Exercises lvalue expressions: array indexing, pointer dereference, comma
   operator. Verifies that removing cmt_lvalue does not break frontend
   handling of these constructs. */
#include <assert.h>

int arr[4] = {10, 20, 30, 40};

int *get_ptr(int *p)
{
  return p;
}

int main()
{
  /* array index lvalue */
  arr[1] = 99;
  assert(arr[1] == 99);

  /* pointer dereference lvalue */
  int *p = &arr[0];
  *p = 7;
  assert(arr[0] == 7);

  /* pointer arithmetic index (p[i] -> *(p+i)) */
  p[2] = 55;
  assert(arr[2] == 55);

  /* comma operator: value is last operand */
  int x = 0, y = 0;
  (x = 1, y = 2);
  assert(x == 1 && y == 2);

  return 0;
}
