// libFuzzer entry point for the real irep_container refcount lifecycle.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "refcount_ops.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
  // abort(), not assert(): fuzz builds may define NDEBUG.
  if (!irep2_refcount_fuzz::run_ops(Data, Size))
    abort();
  return 0;
}
