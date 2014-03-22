#include "yices_conv.h"

smt_convt *
create_new_yices_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                              const optionst &opts __attribute__((unused)),
                              tuple_iface **tuple_api __attribute__((unused)),
                              array_iface **array_api)
{
  yices_convt *conv = new yices_convt(int_encoding, ns, is_cpp);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

yices_convt::yices_convt(bool int_encoding, const namespacet &ns, bool is_cpp)
  : smt_convt(int_encoding, ns, is_cpp), array_iface(false, false)
{
  yices_init();
}

smt_convt::resultt
yices_convt::dec_solve()
{
  abort();
}

tvt
yices_convt::l_get(smt_astt l)
{
  abort();
}

const std::string
yices_convt::solver_text()
{
  abort();
}

void
yices_convt::assert_ast(smt_astt a)
{
  abort();
}

smt_astt
yices_convt::mk_func_app(smt_sortt s, smt_func_kind k,
                             smt_astt const *args,
                             unsigned int numargs)
{
  abort();
}

smt_sortt
yices_convt::mk_sort(const smt_sort_kind k, ...)
{
  abort();
}

smt_astt
yices_convt::mk_smt_int(const mp_integer &theint, bool sign)
{
  abort();
}

smt_astt
yices_convt::mk_smt_real(const std::string &str)
{
  abort();
}

smt_astt
yices_convt::mk_smt_bvint(const mp_integer &theint, bool sign,
                              unsigned int w)
{
  abort();
}

smt_astt
yices_convt::mk_smt_bool(bool val)
{
  abort();
}

smt_astt
yices_convt::mk_smt_symbol(const std::string &name, smt_sortt s)
{
  abort();
}

smt_astt
yices_convt::mk_array_symbol(const std::string &name, smt_sortt s)
{
  abort();
}

smt_sortt
yices_convt::mk_struct_sort(const type2tc &type)
{
  abort();
}

smt_sortt
yices_convt::mk_union_sort(const type2tc &type)
{
  abort();
}

smt_astt
yices_convt::mk_extract(smt_astt a, unsigned int high,
                            unsigned int low, smt_sortt s)
{
  abort();
}

smt_astt
yices_convt::convert_array_of(const expr2tc &init_val,
                                unsigned long domain_width)
{
  abort();
}

void
yices_convt::add_array_constraints_for_solving()
{
  abort();
}

expr2tc
yices_convt::get_bool(smt_astt a)
{
  abort();
}

expr2tc
yices_convt::get_bv(const type2tc &t, smt_astt a)
{
  abort();
}

expr2tc
yices_convt::get_array_elem(smt_astt array, uint64_t index,
                       const type2tc &subtype)
{
  abort();
}
