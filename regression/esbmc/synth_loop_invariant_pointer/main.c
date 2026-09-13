/* Regression for a defect found in review of the invariant synthesiser; see
 * test.desc for the flags. Without the fix this program is mis-verified. */
#include <assert.h>
int main(void) {
  int arr[4] = {0,1,2,3}; int *p = arr; unsigned int i = 0;
  while (i < 3) { p = p + 1; i = i + 1; }
  assert(p == arr);
  return 0;
}
