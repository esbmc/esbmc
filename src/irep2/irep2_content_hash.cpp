#include <irep2/irep2_content_hash.h>

#include <irep2/irep2_dispatch.h>
#include <util/arith/mp_arith.h>

namespace
{
/// Two independently-seeded FNV-1a lanes advanced in one pass: 128 bits, so a
/// key can stand on its own without storing the text it summarises.
struct hashert
{
  uint64_t lo, hi;

  void byte(unsigned char c)
  {
    lo ^= c;
    lo *= 0x100000001b3ULL;
    hi ^= c + 0x9e3779b97f4a7c15ULL;
    hi *= 0xc2b2ae3d27d4eb4fULL;
  }

  void number(uint64_t v)
  {
    for (int i = 0; i < 8; ++i)
      byte(static_cast<unsigned char>(v >> (i * 8)));
  }

  void text(const std::string &s)
  {
    number(s.size());
    for (unsigned char c : s)
      byte(c);
  }
};

struct walkert
{
  hashert &out;
  const irep2_name_mappert &rename;
  const irep2_symbol_mappert &rename_symbol;

  void field(const irep_idt &v)
  {
    out.text(rename(v));
  }
  void field(const std::vector<irep_idt> &v)
  {
    out.number(v.size());
    for (const irep_idt &e : v)
      field(e);
  }
  void field(const BigInt &v)
  {
    out.text(integer2string(v));
  }
  void field(const fixedbvt &v)
  {
    out.text(v.to_ansi_c_string());
  }
  void field(const ieee_floatt &v)
  {
    field(v.pack());
  }
  void field(const expr2tc &v)
  {
    expr(v);
  }
  void field(const type2tc &v)
  {
    type(v);
  }
  void field(const std::vector<expr2tc> &v)
  {
    out.number(v.size());
    for (const expr2tc &e : v)
      expr(e);
  }
  void field(const std::vector<type2tc> &v)
  {
    out.number(v.size());
    for (const type2tc &e : v)
      type(e);
  }
  template <class T>
  void field(const T &v)
  {
    out.number(static_cast<uint64_t>(v));
  }

  template <class K>
  void all_fields(const K &node)
  {
    std::apply([&](auto... mp) { (field(node.*mp), ...); }, K::fields);
  }

  void expr(const expr2tc &e)
  {
    if (!e)
    {
      out.number(0);
      return;
    }

    out.number(1);
    out.number(e->expr_id);
    type(e->type);

    // A symbol's SSA version is an ordinary field (level2_num), so hashing it
    // would pin the digest to the absolute version symex happened to reach.
    if (e->expr_id == expr2t::symbol_id)
    {
      out.text(rename_symbol(static_cast<const symbol2t &>(*e)));
      return;
    }

    switch (e->expr_id)
    {
#define IREP2_EXPR(kind, _)                                                    \
  case expr2t::kind##_id:                                                      \
    all_fields(static_cast<const kind##2t &>(*e));                             \
    return;
#include <irep2/expr_kinds.inc>
#undef IREP2_EXPR
    case expr2t::end_expr_id:
      break;
    }
  }

  void type(const type2tc &t)
  {
    if (!t)
    {
      out.number(0);
      return;
    }

    out.number(1);
    out.number(t->type_id);

    switch (t->type_id)
    {
#define IREP2_TYPE(kind, _)                                                    \
  case type2t::kind##_id:                                                      \
    all_fields(static_cast<const kind##_type2t &>(*t));                        \
    return;
#include <irep2/type_kinds.inc>
#undef IREP2_TYPE
    case type2t::end_type_id:
      break;
    }
  }
};
} // namespace

void irep2_content_hash(
  const expr2tc &e,
  const irep2_name_mappert &rename,
  const irep2_symbol_mappert &rename_symbol,
  uint64_t &lo,
  uint64_t &hi)
{
  hashert h{lo, hi};
  walkert{h, rename, rename_symbol}.expr(e);
  lo = h.lo;
  hi = h.hi;
}
