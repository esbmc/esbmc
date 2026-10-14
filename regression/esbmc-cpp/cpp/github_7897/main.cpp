// #7897: a vector comparison yields a lane mask. The global initialiser binds
// its side-effecting operand at file scope, where there is no enclosing block.
#include <cassert>

typedef float v4f __attribute__((__vector_size__(16)));
typedef int v4i __attribute__((__vector_size__(16)));

static int calls;

static v4f next()
{
  ++calls;
  return v4f{1, 2, 3, 4};
}

v4f a = {1, 2, 3, 4};
v4i global_mask = next() == a;

int main()
{
  assert(calls == 1);
  assert(global_mask[0] == -1 && global_mask[3] == -1);

  v4i lt = a < v4f{2, 2, 2, 2};
  assert(lt[0] == -1 && lt[1] == 0);
  return 0;
}
