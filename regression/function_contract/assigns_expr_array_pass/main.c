/* Phase 2: Expression-based assigns with array element access
 * Tests arr[i].field syntax (not supported in Phase 1)
 */
#include <assert.h>

typedef struct {
    int id;
    int value;
} Node;

Node nodes[3];

void init_node(int idx)
{
  __ESBMC_requires(idx >= 0 && idx < 3);
  
  // Phase 2: Array element field access!
  __ESBMC_assigns(nodes[idx].id, nodes[idx].value);
  
  __ESBMC_ensures(nodes[idx].id == idx);
  __ESBMC_ensures(nodes[idx].value == idx * 10);
  
  nodes[idx].id = idx;
  nodes[idx].value = idx * 10;
}

int main()
{
  init_node(0);
  init_node(1);
  init_node(2);
  
  assert(nodes[0].id == 0 && nodes[0].value == 0);
  assert(nodes[1].id == 1 && nodes[1].value == 10);
  assert(nodes[2].id == 2 && nodes[2].value == 20);
  
  return 0;
}
