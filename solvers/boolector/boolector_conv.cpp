#include "boolector_conv.h"

smt_convt *
create_new_boolector_solver(bool is_cpp, bool int_encoding,
                            const namespacet &ns)
{
  return new boolector_convt(is_cpp, int_encoding, ns);
}

boolector_convt::boolector_convt(bool is_cpp, bool int_encoding,
                                 const namespacet &ns)
  : smt_convt(true, int_encoding, ns, is_cpp, false, false, true)
{
  abort();
}

boolector_convt::~boolector_convt(void)
{
}

smt_convt::resultt
boolector_convt::dec_solve()
{
  abort();
}

tvt
boolector_convt::l_get(const smt_ast *l __attribute__((unused)))
{
  abort();
}

const std::string
boolector_convt::solver_text()
{
  abort();
}

void
boolector_convt::assert_ast(const smt_ast *a __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_func_app(const smt_sort *s __attribute__((unused)), smt_func_kind k __attribute__((unused)),
                               const smt_ast * const *args __attribute__((unused)),
                               unsigned int numargs __attribute__((unused)))
{
  abort();
}

smt_sort *
boolector_convt::mk_sort(const smt_sort_kind k __attribute__((unused)), ...)
{
  abort();
}

smt_ast *
boolector_convt::mk_smt_int(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_smt_bvint(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)), unsigned int w __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_smt_bool(bool val __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_smt_symbol(const std::string &name __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}

smt_sort *
boolector_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_sort *
boolector_convt::mk_union_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_extract(const smt_ast *a __attribute__((unused)), unsigned int high __attribute__((unused)), unsigned int low __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}

expr2tc
boolector_convt::get_bool(const smt_ast *a __attribute__((unused)))
{
  abort();
}

expr2tc
boolector_convt::get_bv(const type2tc &t __attribute__((unused)), const smt_ast *a __attribute__((unused)))
{
  abort();
}

expr2tc
boolector_convt::get_array_elem(const smt_ast *array __attribute__((unused)), uint64_t index __attribute__((unused)), const smt_sort *sort __attribute__((unused)))
{
  abort();
}
