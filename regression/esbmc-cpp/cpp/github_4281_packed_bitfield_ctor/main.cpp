// Issue #4281: false-positive bounds check on a [[gnu::packed]] struct
// whose constructor initialises a bitfield member via the
// member-initialiser-list. This must verify successfully.
#include <cstdint>

struct [[gnu::packed]] S
{
  uint32_t flag : 1;
  uint32_t rest : 31;
  S() : flag{0}, rest{0}
  {
  }
};

int main()
{
  S s;
  (void)s.flag;
  return 0;
}
