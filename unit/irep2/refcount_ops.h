// Shared driver: decode a byte stream into container operations on N real
// expr2tc slots and check that each live node's refcount equals its
// live-handle count (I1). Used by refcount.fuzz.cpp and refcount.test.cpp.

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
inline bool conservation_holds(const std::vector<expr2tc> &slots)
{
  for (const expr2tc &s : slots)
  {
    if (!s)
      continue;
    const expr2t *node = s.get(); // const get(): does not detach/perturb
    unsigned live = 0;
    for (const expr2tc &t : slots)
      if (t && t.get() == node)
        ++live;
    if (node->refcount.load(std::memory_order_relaxed) != live)
      return false;
  }
  return true;
}

// Returns false iff a step leaves some node's refcount != its live-handle count.
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
    case 0: // make
      slots[i] = gen_ulong(arg);
      break;
    case 1: // copy-assign
      slots[i] = slots[j];
      break;
    case 2: // move-assign
      slots[i] = std::move(slots[j]);
      break;
    case 3: // copy-ctor then assign
    {
      expr2tc tmp(slots[j]);
      slots[i] = tmp;
      break;
    }
    case 4: // detach (non-const get)
      if (slots[i])
        (void)slots[i].get();
      break;
    case 5: // reset
      slots[i] = expr2tc();
      break;
    case 6: // swap
    {
      using std::swap;
      swap(slots[i], slots[j]);
      break;
    }
    default:
      slots[i].reset();
      break;
    }

    if (!conservation_holds(slots))
      return false;
  }
  return true;
}
} // namespace irep2_refcount_fuzz
