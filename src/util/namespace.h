#ifndef CPROVER_NAMESPACE_H
#define CPROVER_NAMESPACE_H

#include <util/context.h>
#include <irep2/irep2.h>
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
    typet back = migrate_type_back(src);
    typet followed = follow(back);
    type2tc tmp = migrate_type(followed);
    return tmp;
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
