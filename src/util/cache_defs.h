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
    const expr2tc &e1 = p.first;
    const expr2tc &e2 = p.second;
    crypto_hash h1, h2;
    e1->hash(h1);
    h1.fin();
    e2->hash(h2);
    h2.fin();
    return h1.to_size_t() ^ h2.to_size_t();
  }
};
} // namespace std
