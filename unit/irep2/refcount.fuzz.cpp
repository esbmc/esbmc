// libFuzzer entry point: drive the REAL irep_container<T> refcount lifecycle
// with nondeterministic input. libFuzzer supplies the byte stream; run_ops
// decodes it into a sequence of container operations over real expr2tc slots
// and checks refcount conservation (I1) after every step, while ASan
// (-fsanitize=fuzzer,address) catches any use-after-free / double-free.
//
// Built only when -DENABLE_FUZZER=On (see unit/irep2/CMakeLists.txt). Run with:
//   ./unit/irep2/irep2refcountfuzz -runs=1000000
// The deterministic Catch2 replay in refcount.test.cpp exercises the same
// run_ops driver in normal CI so the property is pinned even with fuzzing off.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "refcount_ops.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
  // Explicit abort (not assert): fuzz builds usually define NDEBUG, which
  // would compile out an assert and silently disable the conservation
  // oracle. ASan (-fsanitize=fuzzer,address) independently flags any
  // use-after-free / double-free in the real container operations.
  if (!irep2_refcount_fuzz::run_ops(Data, Size))
    abort();
  return 0;
}
