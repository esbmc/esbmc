/* clang 15+ promotes -Wint-conversion to a hard error; GCC still accepts it.
 * Preprocessed kernel and CIL sources write sentinel pointers as integer
 * constants, so erroring out rejected input a mainstream toolchain compiles --
 * a false PARSING ERROR. The suppression used to be gated behind --sv-comp,
 * which meant an ordinary run refused these sources. */
#include <assert.h>

struct lock
{
  void *owner;
  unsigned magic;
};

static struct lock l = {0xffffffffffffffffUL, 0xdead4ead};

int main(void)
{
  assert(l.magic == 0xdead4ead);
  return 0;
}
