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

  // Type and value accessors. After B2 S5a (#4735, lazy variant restored in
  // #4739) and B2 V2 (esbmc/esbmc#4715, the value-side flip below) both
  // representations are *caches* sharing the same storage layout: whichever
  // setter last wrote marks its side valid and invalidates the other; the
  // const reader on the invalidated side lazily derives via the migration
  // layer on first access.
  //
  // The lazy split is deliberate: an eager forward migrate at set_type /
  // set_value would walk legacy sub-expressions that the existing frontends
  // sometimes build without populating intermediate types or kinds. The
  // post-#4739 design tolerates those latent holes as long as nothing reads
  // the IREP2 side of the affected symbol -- which, in practice, no current
  // pipeline path does for the tmp symbols involved.
  const typet &get_type() const;
  const exprt &get_value() const;
  const type2tc &get_type2() const;
  const expr2tc &get_value2() const;

  // Type setters. Each writes one side and invalidates the other; the read
  // side derives lazily on first access (with the nil/empty-id case guarded
  // inside the readers).
  void set_type(const typet &t);
  void set_type(typet &&t);
  void set_type(const type2tc &t);

  // Value setters. Mirror of the type setters after B2 V2: legacy variants
  // store the legacy form and invalidate the IREP2 side; the IREP2-side
  // setter stores expr2tc directly and invalidates the legacy side. V1
  // (#4737) is the precondition that makes the back-migration safe for
  // function-body symbols (code_block2t round-trips).
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
  // Type: both representations are caches sharing the S5a storage layout.
  // The setter that wrote last marks its side valid and invalidates the
  // other. Reading the other side lazily derives via migrate_type /
  // migrate_type_back. Both fields are mutable so the const getters can
  // populate from the other side on demand.
  mutable type2tc type_;
  mutable typet legacy_type_cache_;
  mutable bool legacy_type_valid_;
  mutable bool type2_valid_;

  // Value: same shape as the type side after B2 V2. expr2tc is the dominant
  // form on IREP2-side writes; legacy `exprt` is derived lazily via
  // migrate_expr_back. V1 (#4737) closed the back-migration coverage gap so
  // function bodies (code_block2t and the four other previously-missing
  // kinds) can round-trip, which is the precondition for the V2 flip to be
  // safe.
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
