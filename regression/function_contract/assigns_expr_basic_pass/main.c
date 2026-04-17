/* Phase 2: Expression-based __ESBMC_assigns test
 * Tests that assigns with expression arguments (not strings) works correctly
 */
#include <assert.h>

int global_x = 10;
int global_y = 20;
int global_z = 30;

void modify_x()
{
  // Phase 2: Use expression, not string!
  __ESBMC_assigns(global_x);
  __ESBMC_ensures(global_x == 11);
  
  global_x = 11;
}

int main()
{
  modify_x();
  
  // global_x should be modified
  assert(global_x == 11);
  
  // global_y and global_z should NOT be havoc'd (precise havoc)
  assert(global_y == 20);
  assert(global_z == 30);
  
  return 0;
}
