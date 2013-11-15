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
  if (int_encoding) {
    std::cerr << "Boolector does not support integer encoding mode"<< std::endl;
    abort();
  }

  btor = boolector_new();
  boolector_enable_model_gen(btor);
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
  return "Boolector";
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
boolector_convt::mk_sort(const smt_sort_kind k, ...)
{
  // Boolector doesn't have any special handling for sorts, they're all always
  // explicit arguments to functions. So, just use the base smt_sort class.
  va_list ap;
  smt_sort *s = NULL, *dom, *range;
  unsigned long uint;
  bool thebool;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
  case SMT_SORT_REAL:
    std::cerr << "Boolector does not support integer encoding mode"<< std::endl;
    abort();
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    thebool = thebool;
    s = new smt_sort(k, uint);
    break;
  case SMT_SORT_ARRAY:
    dom = va_arg(ap, smt_sort *); // Consider constness?
    range = va_arg(ap, smt_sort *);
    s = new smt_sort(k, range->data_width, dom->data_width);
    break;
  case SMT_SORT_BOOL:
    s = new smt_sort(k);
    break;
  default:
    std::cerr << "Unhandled SMT sort in boolector conv" << std::endl;
    abort();
  }

  return s;
}

smt_ast *
boolector_convt::mk_smt_int(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)))
{
  std::cerr << "Boolector can't create integer sorts" << std::endl;
  abort();
}

smt_ast *
boolector_convt::mk_smt_real(const std::string &str __attribute__((unused)))
{
  std::cerr << "Boolector can't create Real sorts" << std::endl;
  abort();
}

smt_ast *
boolector_convt::mk_smt_bvint(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)), unsigned int w __attribute__((unused)))
{
  abort();
}

smt_ast *
boolector_convt::mk_smt_bool(bool val)
{
  BtorNode *node = (val) ? boolector_true(btor) : boolector_false(btor);
  const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
  return new btor_smt_ast(sort, node);
}

smt_ast *
boolector_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  symtable_type::iterator it = symtable.find(name);
  if (it != symtable.end())
    return it->second;

  BtorNode *node;

  switch(s->id) {
  case SMT_SORT_BV:
    node = boolector_var(btor, s->data_width, name.c_str());
    break;
  case SMT_SORT_BOOL:
    node = boolector_var(btor, 1, name.c_str());
    break;
  case SMT_SORT_ARRAY:
    node = boolector_array(btor, s->data_width, s->domain_width, name.c_str());
    break;
  default:
    return NULL; // Hax.
  }

  btor_smt_ast *ast = new btor_smt_ast(s, node);

  symtable.insert(symtable_type::value_type(name, ast));
  return ast;
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
