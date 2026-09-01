/* Regression for a defect found in review of the invariant synthesiser; see
 * test.desc for the flags. Without the fix this program is mis-verified. */
#include <assert.h>
int main(void) { int n; int i = 0; while (i < n) i++; assert(i == n || n <= 0); return 0; }
