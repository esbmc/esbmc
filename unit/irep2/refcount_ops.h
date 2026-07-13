// Nondeterministic-input operation harness for the real irep_container<T>
// refcount lifecycle (H-A1, shared by the libFuzzer target refcount.fuzz.cpp
// and the deterministic Catch2 replay in refcount.test.cpp).
//
// A byte stream (from libFuzzer, or a fixed-seed generator in the unit test)
// is decoded into a sequence of container operations — make / copy-assign /
// copy-ctor / move / detach / reset / swap — over N real expr2tc slots. After
// every operation the harness applies a *direct differential oracle* against
// the genuine implementation: for each live node, the real irep2t::refcount
// atomic must equal the number of slots that actually point at that node
// (invariant I1). Reading identity/refcount goes through the CONST get()
// overload so the check never triggers detach() and perturbs what it measures.
//
// The refcount miscount is caught by this oracle on any build; use-after-free
// / double-free are caught by ASan when built as the libFuzzer target
// (-fsanitize=fuzzer,address) or under a `Sanitizer` build.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/config.h>

namespace irep2_refcount_fuzz
{
// I1: every live node's real refcount == number of slots pointing at it.
inline bool conservation_holds(const std::vector<expr2tc> &slots)
{
  for (const expr2tc &s : slots)
  {
    if (!s)
      continue;
    const expr2t *node = s.get(); // const get(): no detach, no perturbation
    unsigned live = 0;
    for (const expr2tc &t : slots)
      if (t && t.get() == node)
        ++live;
    if (node->refcount.load(std::memory_order_relaxed) != live)
      return false;
  }
  return true;
}

// Decode `data` as an opcode stream and drive N real containers, checking
// refcount conservation after each step. Returns false iff the oracle is
// ever violated (the real refcount disagrees with the live-handle count).
inline bool run_ops(const uint8_t *data, size_t size, size_t slot_count = 5)
{
  config.ansi_c.word_size = 32;
  std::vector<expr2tc> slots(slot_count);

  for (size_t pos = 0; pos + 1 < size; pos += 2)
  {
    const uint8_t sel = data[pos];
    const uint8_t arg = data[pos + 1];
    const unsigned op = sel & 0x7;
    const size_t i = (sel >> 3) % slot_count;
    const size_t j = arg % slot_count;

    switch (op)
    {
    case 0: // make a fresh node (refcount 1)
      slots[i] = gen_ulong(arg);
      break;
    case 1: // copy-assign: release old, fetch_add on source
      slots[i] = slots[j];
      break;
    case 2: // move-assign: steal, no bump (self-move is guarded)
      slots[i] = std::move(slots[j]);
      break;
    case 3: // copy-ctor into a temporary, then assign it in
    {
      expr2tc tmp(slots[j]);
      slots[i] = tmp;
      break;
    }
    case 4: // detach: non-const get() clones iff shared
      if (slots[i])
        (void)slots[i].get();
      break;
    case 5: // reset via assignment of an empty container
      slots[i] = expr2tc();
      break;
    case 6: // swap two slots
    {
      using std::swap;
      swap(slots[i], slots[j]);
      break;
    }
    default: // reset via reset()
      slots[i].reset();
      break;
    }

    if (!conservation_holds(slots))
      return false;
  }
  return true;
}
} // namespace irep2_refcount_fuzz
