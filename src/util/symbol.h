#ifndef CPROVER_SYMBOL_H
#define CPROVER_SYMBOL_H

#include <algorithm>

#include <list>
#include <vector>
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
  // every caller again. The mutable overloads are a transitional convenience
  // (writes/swaps); a later step routes writes through a setter so the IREP2
  // form can be kept in sync.
  const typet &get_type() const
  {
    return type;
  }
  typet &get_type()
  {
    return type;
  }
  const exprt &get_value() const
  {
    return value;
  }
  exprt &get_value()
  {
    return value;
  }

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
