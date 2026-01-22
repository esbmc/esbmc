/* Phase 2: Test that weak ensures leads to verification failure
 * This should FAIL because ensures doesn't constrain all havoc'd variables
 */
#include <assert.h>

int global_x = 10;
int global_y = 20;

void weak_contract()
{
  // Assigns both x and y
  __ESBMC_assigns(global_x, global_y);
  
  // BUG: ensures only constrains x, not y!
  __ESBMC_ensures(global_x == 11);
  
  global_x = 11;
  global_y = 21;
}

int main()
{
  weak_contract();
  
  // This assertion should fail because:
  // - global_y is in assigns (will be havoc'd)
  // - ensures doesn't constrain global_y
  // - So global_y could be any value, not necessarily 21
  assert(global_y == 21);
  
  return 0;
}
