/* Phase 2: Expression-based assigns with pointer field access
 * Tests node->field syntax (not supported in Phase 1)
 */
#include <assert.h>
#include <stddef.h>

typedef struct {
    int id;
    int value;
    int data;
} Node;

void update_value(Node *node, int new_val)
{
  __ESBMC_requires(node != NULL);
  
  // Phase 2: Pointer field access expression!
  __ESBMC_assigns(node->value);
  
  __ESBMC_ensures(node->value == new_val);
  __ESBMC_ensures(node->id == __ESBMC_old(node->id));
  __ESBMC_ensures(node->data == __ESBMC_old(node->data));
  
  node->value = new_val;
}

int main()
{
  Node n;
  n.id = 1;
  n.value = 10;
  n.data = 100;
  
  update_value(&n, 999);
  
  assert(n.id == 1);      // Should not change
  assert(n.value == 999); // Should change
  assert(n.data == 100);  // Should not change
  
  return 0;
}
