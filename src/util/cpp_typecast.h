#ifndef ESBMC_CPP_TYPECAST_H
#define ESBMC_CPP_TYPECAST_H

#include "expr.h"
#include "namespace.h"

class cpp_typecastt
{
public:
  explicit cpp_typecastt(const namespacet &_ns) : ns(_ns)
  {
  }

  virtual ~cpp_typecastt() = default;

  virtual void
  derived_to_base_typecast(exprt &expr, const typet &type, bool is_virtual);

protected:
  const namespacet &ns;
  void adjust_pointer_offset(
    exprt &expr,
    const typet &src_type,
    const typet &dest_type,
    bool is_virtual);
  bool try_non_virtual_cast(
    exprt &expr,
    const typet &dest_type,
    const dstring &dest_sub_name,
    const typet &src_type) const;
  bool try_virtual_cast(
    exprt &expr,
    const typet &dest_type,
    const dstring &dest_sub_name,
    const typet &src_type) ;
};
#endif //ESBMC_CPP_TYPECAST_H
