#pragma once
#include <irep2/irep2.h>

// This header will prevent the dependency hell
// between irep2, message and config

using assert_pair = std::pair<expr2tc, expr2tc>;
using assert_db = std::unordered_set<assert_pair>;

namespace std
{
template <>
struct hash<assert_pair>
{
  auto operator()(const assert_pair &p) const -> size_t
  {
    // crc() is irep2's cached structural hash (a size_t). The previous code
    // built two full SHA-1s only to fold each down to a size_t via
    // to_size_t() — same width, so crc() is lossless here and skips the
    // per-byte SHA walk.
    return p.first->crc() ^ p.second->crc();
  }
};
} // namespace std
