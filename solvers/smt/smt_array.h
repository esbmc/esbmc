#include "smt_conv.h"

// Interface definition for array manipulation

class array_iface
{
  virtual smt_astt mk_array_symbol(const std::string &name, smt_sortt sort) = 0;
  virtual expr2tc get_array_elem(smt_astt a, uint64_t idx,
                                 const type2tc &subtype) = 0;
  virtual const smt_ast *convert_array_of(const expr2tc &init_val,
                                           unsigned long domain_width) = 0;
  // And everything else goes through the ast methods!
};
