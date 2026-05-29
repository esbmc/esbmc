#ifndef CPROVER_NAMESPACE_H
#define CPROVER_NAMESPACE_H

#include <util/context.h>
#include <irep2/irep2.h>
#include <irep2/irep2_type.h>
#include <util/migrate.h>

class namespacet
{
public:
  virtual const symbolt *lookup(const irep_idt &name) const;
  const symbolt *lookup(const irept &irep) const
  {
    return lookup(irep.identifier());
  }

  void follow_symbol(irept &irep) const;

  const typet &follow(const typet &src) const;
  const type2tc follow(const type2tc &src) const
  {
    // Native IREP2 symbol-type resolution, mirroring follow(typet) without
    // the back-migrate -> follow(typet) -> forward-migrate detour (hot path).
    if (!is_symbol_type(src))
      return src;

    const symbolt *symbol = lookup(to_symbol_type(src).symbol_name);

    // let's hope it's not cyclic...
    while (true)
    {
      assert(symbol);
      assert(symbol->is_type);
      type2tc t = migrate_symbol_type(*symbol);
      if (!is_symbol_type(t))
        return t;
      const symbolt *next = lookup(to_symbol_type(t).symbol_name);
      assert(next != symbol && "cycle of length 1 in ns.follow()");
      symbol = next;
    }
  }

  namespacet() = delete;

  virtual ~namespacet() = default;

  namespacet(const contextt &_context) : context(&_context)
  {
  }

  virtual unsigned get_max(const std::string &prefix) const;

  const contextt &get_context() const
  {
    return *context;
  }

protected:
  const contextt *context;
};

#endif
