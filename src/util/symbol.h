#ifndef CPROVER_SYMBOL_H
#define CPROVER_SYMBOL_H

#include <list>
#include <vector>
#include <irep2/irep2.h>
#include <util/config.h>
#include <util/expr.h>
#include <util/location.h>

class symbolt
{
public:
  locationt location;
  irep_idt id;
  irep_idt module;
  irep_idt name;
  irep_idt mode;

  // global use
  bool is_type, is_macro, is_parameter;

  // ANSI-C
  bool lvalue, static_lifetime, file_local, is_extern, is_thread_local;

  // For python use
  bool is_set;
  std::vector<typet> python_annotation_types;

  symbolt();

  // Type accessors. After the B2 S5a source-of-truth flip
  // (esbmc/esbmc#4715), the IREP2 form is the stored field; the legacy
  // `typet` is derived lazily via migrate_type_back and cached.
  //
  //   get_type()  - lazy legacy cache populated from the IREP2 source on
  //                 first read; invalidated by any IREP2-side write.
  //   get_type2() - the IREP2 source of truth directly (O(1), no cache
  //                 logic on this side).
  //
  // Value accessors keep the dual-storage shape that S4b shipped: legacy
  // authoritative, IREP2 lazy cache. The value-side flip is out of scope
  // for Phase 5 (see docs/irep2-symbol-table-phase5-plan.md V-track).
  const typet &get_type() const;
  const exprt &get_value() const
  {
    return value;
  }
  const type2tc &get_type2() const
  {
    return type_;
  }
  const expr2tc &get_value2() const;

  // Type setters. The legacy setter forward-migrates to the IREP2 source
  // and populates the legacy cache eagerly with its input (no
  // back-migration needed for the next get_type() read). The IREP2 setter
  // stores `t` as the source and invalidates the legacy cache; the next
  // get_type() back-migrates (with the nil case handled explicitly).
  void set_type(const typet &t);
  void set_type(typet &&t);
  void set_type(const type2tc &t);

  // Value setters keep the S4b semantics: write legacy, invalidate the
  // IREP2 lazy cache.
  void set_value(const exprt &v);
  void set_value(exprt &&v);

  void clear();

  void swap(symbolt &b);

  void show(std::ostream &out) const;
  DUMP_METHOD void dump() const;

  void to_irep(irept &dest) const;
  void from_irep(const irept &src);

  irep_idt get_function_name() const;

private:
  // Type: IREP2 is the source of truth (B2 S5a). The legacy field is a
  // lazy cache derived via migrate_type_back; mutable so get_type() can
  // populate it from a const method. Consistency is guaranteed by the
  // public setter contract.
  type2tc type_;
  mutable typet legacy_type_cache_;
  mutable bool legacy_type_valid_;

  // Value: legacy authoritative; IREP2 is a lazy cache (S4b shape,
  // unchanged by Phase 5). Function bodies cannot round-trip through
  // migrate_expr_back today, which is why the value side is deliberately
  // not flipped - see docs/irep2-symbol-table-phase5-plan.md for the
  // V-track.
  exprt value;
  mutable expr2tc value2_cache;
  mutable bool value2_valid;
};

std::ostream &operator<<(std::ostream &out, const symbolt &symbol);

typedef std::list<symbolt *> symbol_listt;

#define forall_symbol_list(it, expr)                                           \
  for (symbol_listt::const_iterator it = (expr).begin(); it != (expr).end();   \
       it++)

#define Forall_symbol_list(it, expr)                                           \
  for (symbol_listt::iterator it = (expr).begin(); it != (expr).end(); it++)

typedef std::list<const symbolt *> symbolptr_listt;

#define forall_symbolptr_list(it, list)                                        \
  for (symbolptr_listt::const_iterator it = (list).begin();                    \
       it != (list).end();                                                     \
       it++)

#define Forall_symbolptr_list(it, list)                                        \
  for (symbolptr_listt::iterator it = (list).begin(); it != (list).end(); it++)

#endif
