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

  // Type and value accessors. After the symbol-table migration
  // (esbmc/esbmc#4715, boundary B2), the IREP2 forms `type_` / `value_` are
  // the source of truth on both sides; the legacy `typet` / `exprt` are
  // permanent on-demand caches populated lazily by the const readers.
  // Whichever setter last wrote marks its side valid and invalidates the
  // other; the reader on the invalidated side lazily derives via the
  // migration layer on first access. This is the end-state design, not
  // transitional -- see docs/irep2-migration.md.
  //
  // The lazy split is deliberate: an eager forward migrate at set_type /
  // set_value would walk legacy sub-expressions that the existing frontends
  // sometimes build without populating intermediate types or kinds. The
  // lazy design tolerates those latent holes as long as nothing reads the
  // IREP2 side of the affected symbol -- which, in practice, no current
  // pipeline path does for the tmp symbols involved.
  const typet &get_type() const;
  const exprt &get_value() const;
  const type2tc &get_type2() const;
  const expr2tc &get_value2() const;

  // True iff the IREP2 value is the authoritative form (written via the
  // expr2tc setter), so get_value2() returns it directly without a lazy
  // migration from the legacy exprt. Lets a pass honour the lazy split (see
  // the class comment above): reading the IREP2 side of a legacy-valued symbol
  // would force-migrate sub-expressions the frontend left with unresolved tags
  // -- a latent hole the design tolerates only while nothing reads that side.
  bool has_native_value2() const
  {
    return value2_valid_;
  }

  // Type setters. Each writes one side and invalidates the other; the read
  // side derives lazily on first access (with the nil/empty-id case guarded
  // inside the readers).
  void set_type(const typet &t);
  void set_type(typet &&t);
  void set_type(const type2tc &t);

  // Value setters. Mirror of the type setters: legacy variants store the
  // legacy form and invalidate the IREP2 side; the IREP2-side setter
  // stores expr2tc directly and invalidates the legacy side. `migrate_expr_back`
  // covers every expr2t kind a symbol value may hold -- including
  // code_block2t for function bodies -- so the lazy back-migration in
  // get_value() is safe regardless of what was written.
  void set_value(const exprt &v);
  void set_value(exprt &&v);
  void set_value(const expr2tc &v);

  void clear();

  void swap(symbolt &b);

  void show(std::ostream &out) const;
  DUMP_METHOD void dump() const;

  void to_irep(irept &dest) const;
  void from_irep(const irept &src);

  irep_idt get_function_name() const;

private:
  // Type and value storage. IREP2 (`type_`, `value_`) is the source of
  // truth on both sides; the legacy fields (`legacy_*_cache_`) are
  // permanent on-demand caches. The setter that wrote last marks its
  // side valid and invalidates the other; reading the invalidated side
  // lazily derives via migrate_type / migrate_type_back /
  // migrate_expr / migrate_expr_back. All fields are mutable so the
  // const readers can populate from the other side on demand.
  mutable type2tc type_;
  mutable typet legacy_type_cache_;
  mutable bool legacy_type_valid_;
  mutable bool type2_valid_;

  mutable expr2tc value_;
  mutable exprt legacy_value_cache_;
  mutable bool legacy_value_valid_;
  mutable bool value2_valid_;
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
