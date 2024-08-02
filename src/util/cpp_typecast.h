#ifndef ESBMC_CPP_TYPECAST_H
#define ESBMC_CPP_TYPECAST_H

#include "util/std_types.h"
#include "util/expr.h"
#include "util/namespace.h"
class cpp_typecast
{
public:
  static void derived_to_base_typecast(
    exprt &expr,
    const typet &dest_type,
    bool is_virtual,
    namespacet &ns);

protected:
  static void get_vbot_binding_expr_base(exprt &base, exprt &new_expr);
  static void adjust_pointer_offset(
    exprt &expr,
    const typet &src_type,
    const typet &dest_type,
    bool is_virtual,
    namespacet &ns);
  static bool try_virtual_cast(
    exprt &expr,
    const typet &dest_type,
    const dstring &dest_sub_name,
    const typet &src_type);
  static bool try_non_virtual_cast(
    exprt &expr,
    const typet &dest_type,
    const dstring &dest_sub_name,
    const typet &src_type,
    namespacet &ns);
  static void
  get_id_name(const typet &type, std::string &base_id, std::string &base_name);
};

#endif //ESBMC_CPP_TYPECAST_H
