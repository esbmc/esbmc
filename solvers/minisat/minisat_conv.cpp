#include "minisat_conv.h"

minisat_convt::minisat_convt(bool int_encoding, const namespacet &_ns,
                             bool is_cpp, const optionst &_opts)
         : smt_convt(true, int_encoding, _ns, is_cpp, false, true, true),
           solver(), options(_opts)
{
  
}

minisat_convt::~minisat_convt(void)
{
}

prop_convt::resultt
minisat_convt::dec_solve()
{
  abort();
}

expr2tc
minisat_convt::get(const expr2tc &expr __attribute__((unused)))
{
  abort();
}

const std::string
minisat_convt::solver_text()
{
  abort();
}

tvt
minisat_convt::l_get(literalt l __attribute__((unused)))
{
  abort();
}

void
minisat_convt::assert_lit(const literalt &l __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_func_app(const smt_sort *ressort __attribute__((unused)),
    smt_func_kind f __attribute__((unused)),
    const smt_ast* const* args __attribute__((unused)),
    unsigned int num __attribute__((unused)))
{
  abort();
}

smt_sort*
minisat_convt::mk_sort(smt_sort_kind k, ...)
{
  va_list ap;
  minisat_smt_sort *s = NULL, *dom, *range;
  unsigned long uint;
  int thebool;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    std::cerr << "Can't make Int sorts in Minisat" << std::endl;
    abort();
  case SMT_SORT_REAL:
    std::cerr << "Can't make Real sorts in Minisat" << std::endl;
    abort();
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    s = new minisat_smt_sort(k, uint);
    break;
  case SMT_SORT_ARRAY:
    dom = va_arg(ap, minisat_smt_sort *); // Consider constness?
    range = va_arg(ap, minisat_smt_sort *);
    s = new minisat_smt_sort(k, dom->width, range->width);
    break;
  case SMT_SORT_BOOL:
    s = new minisat_smt_sort(k);
    break;
  default:
    std::cerr << "Unimplemented SMT sort " << k << " in Minisat conversion"
              << std::endl;
    abort();
  }

  return s;
}

literalt
minisat_convt::mk_lit(const smt_ast *val __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_smt_int(const mp_integer &intval __attribute__((unused)), bool sign __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_smt_real(const std::string &value __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_smt_bvint(const mp_integer &inval __attribute__((unused)),
                            bool sign __attribute__((unused)),
                            unsigned int w __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_smt_bool(bool boolval __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_smt_symbol(const std::string &name __attribute__((unused)),
                             const smt_sort *sort __attribute__((unused)))
{
  abort();
}

smt_sort*
minisat_convt::mk_struct_sort(const type2tc &t __attribute__((unused)))
{
  abort();
}

smt_sort*
minisat_convt::mk_union_sort(const type2tc &t __attribute__((unused)))
{
  abort();
}

smt_ast*
minisat_convt::mk_extract(const smt_ast *src __attribute__((unused)), unsigned int high __attribute__((unused)), unsigned int low __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}
