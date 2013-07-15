#include "mathsat_conv.h"

prop_convt *
create_new_mathsat_solver(bool int_encoding, bool is_cpp, const namespacet &ns)
{
    return new mathsat_convt(is_cpp, int_encoding, ns);
}

mathsat_convt::mathsat_convt(bool is_cpp, bool int_encoding,
                             const namespacet &ns)
  : smt_convt(true, int_encoding, ns, is_cpp, false, true, true)
{
  abort();
}

mathsat_convt::~mathsat_convt(void)
{
}

void
mathsat_convt::assert_lit(const literalt &l __attribute__((unused)))
{
  abort();
}

prop_convt::resultt
mathsat_convt::dec_solve()
{
  abort();
}

expr2tc
mathsat_convt::get(const expr2tc &expr __attribute__((unused)))
{
  abort();
}

tvt
mathsat_convt::l_get(literalt l __attribute__((unused)))
{
  abort();
}

const std::string
mathsat_convt::solver_text()
{
  abort();
}

smt_ast *
mathsat_convt::mk_func_app(const smt_sort *s __attribute__((unused)), smt_func_kind k __attribute__((unused)),
                             const smt_ast * const *args __attribute__((unused)),
                             unsigned int numargs __attribute__((unused)))
{
  abort();
}

smt_sort *
mathsat_convt::mk_sort(const smt_sort_kind k __attribute__((unused)), ...)
{
  abort();
}

literalt
mathsat_convt::mk_lit(const smt_ast *s __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_int(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_bvint(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)),
                              unsigned int w __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_bool(bool val __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_smt_symbol(const std::string &name __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}

smt_sort *
mathsat_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_sort *
mathsat_convt::mk_union_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
mathsat_convt::mk_extract(const smt_ast *a __attribute__((unused)), unsigned int high __attribute__((unused)),
                            unsigned int low __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}
