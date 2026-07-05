// Dead code after an unconditional throw makes the try's normal-completion path
// unreachable, so remove_unreachable prunes the try's empty-CATCH pop, leaving
// the region unbalanced. The lowering rebalances the unclosed region before
// recovering it (#5075). The throw is caught, so this verifies SUCCESSFUL.
#include <cassert>

int main()
{
  int x = 0;
  try
  {
    throw 1;
    x = 5; // unreachable: the throw always fires
  }
  catch (int)
  {
    x = 7;
  }
  assert(x == 7);
  return 0;
}
