#ifndef CPROVER_SYMBOL_H
#define CPROVER_SYMBOL_H

#include <algorithm>

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

  // Accessors for the (now private) `type`/`value` fields. Introduced for the
  // IREP2 symbol-table migration (esbmc/esbmc#4715, B2): all access goes
  // through these so the storage can later become IREP2-native without touching
  // every caller again.
  const typet &get_type() const
  {
    return type;
  }
  const exprt &get_value() const
  {
    return value;
  }

  // Setters for the (private) type/value fields. Writes go exclusively through
  // these (audit completed in B2 S4a, esbmc/esbmc#4715) so the IREP2 shadow
  // (S4b) can be kept consistent: every write invalidates the cache; the next
  // get_type2/get_value2 lazily re-derives via migrate_type/migrate_expr.
  void set_type(const typet &t);
  void set_type(typet &&t);
  void set_value(const exprt &v);
  void set_value(exprt &&v);

  // IREP2-side accessors (esbmc/esbmc#4715, B2 S4b). The IREP2 form of the
  // type/value is cached: populated lazily by these getters via
  // migrate_type/migrate_expr, eagerly by set_type(const type2tc&), and
  // invalidated by every legacy setter. migrate_symbol_type /
  // migrate_symbol_value read these directly, replacing the previous O(size)
  // re-migration on every read with an O(1) cached lookup.
  const type2tc &get_type2() const;
  const expr2tc &get_value2() const;

  // IREP2-side type setter (S4b). Stores the IREP2 form authoritatively and
  // derives the legacy `typet` once via migrate_type_back, instead of the
  // legacy-first / re-migrate-on-read sequence in set_symbol_type. Currently
  // called via set_symbol_type(symbolt&, const type2tc&) in util/migrate.h;
  // exposed here so a future flip can route directly. No expr2tc-side setter:
  // function-body values cannot round-trip (migrate_expr_back rejects
  // code_block2t, see unit/util/migrate.test.cpp), and no caller writes IREP2
  // symbol values today.
  void set_type(const type2tc &t);

  void clear();

  void swap(symbolt &b);

  void show(std::ostream &out) const;
  DUMP_METHOD void dump() const;

  void to_irep(irept &dest) const;
  void from_irep(const irept &src);

  irep_idt get_function_name() const;

private:
  typet type;
  exprt value;

  // IREP2 shadow of `type` / `value`. `*_valid` distinguishes
  // "cached form matches legacy" from "stale, recompute on next read". The
  // shadow is mutable so get_type2/get_value2 can populate it from a const
  // method; consistency with legacy is ensured by the public setter contract
  // (S4a routed every write through set_type/set_value, S4b makes them invalidate).
  mutable type2tc type2_cache;
  mutable expr2tc value2_cache;
  mutable bool type2_valid;
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
