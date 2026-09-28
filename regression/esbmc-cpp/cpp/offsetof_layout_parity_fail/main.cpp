// Anti-vacuity twin of offsetof_layout_parity: both spellings have to compute a
// real offset, so comparing a member against the wrong one must fail.
#include <cstddef>
#include <cassert>

#define ESB(t, m) ((std::size_t)__ESBMC_POINTER_OFFSET(&((t *)0)->m))
#define CLG(t, m) offsetof(t, m)

struct plain
{
  char a;
  int b;
  double c;
};

int main()
{
  assert(ESB(struct plain, b) == CLG(struct plain, c));
  return 0;
}
