#include <assert.h>

/* Anonymous records are named after their source location. Two of them
 * expanded from a single macro share that location, which previously made the
 * inner type resolve to the outer one and sent padding computation into
 * unbounded recursion (stack overflow). Mirrors Linux's
 * __DECLARE_FLEX_ARRAY() in include/uapi/linux/stddef.h. */
#define FLEX(TYPE, NAME)                                                       \
  struct                                                                       \
  {                                                                            \
    struct                                                                     \
    {                                                                          \
    } __empty_##NAME;                                                          \
    TYPE NAME[];                                                               \
  }

struct outer
{
  int n;
  union
  {
    int single;
    FLEX(int, many);
  };
};

/* A second expansion must stay distinct from the first. */
struct other
{
  union
  {
    long single;
    FLEX(long, many);
  };
};

int main(void)
{
  struct outer o;
  o.n = 3;
  o.single = 7;
  assert(o.n == 3);
  assert(o.single == 7);

  struct other t;
  t.single = 9;
  assert(t.single == 9);
  return 0;
}
