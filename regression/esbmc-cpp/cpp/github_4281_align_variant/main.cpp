// Issue #4281 (Context section): the OpenSMA-style packed struct with a
// leading byte field followed by bitfields used to trip a misaligned-access
// false positive on the same member-initialiser-list code path.
#include <cstdint>

struct [[gnu::packed]] R
{
  uint8_t mode;
  uint32_t persistent_data_modified : 1;
  uint32_t other_flags : 31;
  R() : mode{0}, persistent_data_modified{0}, other_flags{0}
  {
  }
};

int main()
{
  R r;
  (void)r.mode;
  (void)r.persistent_data_modified;
  return 0;
}
