#include "minisat_conv.h"

// Utility functions -- echoing CBMC quite a bit. The plan is to build up
// what's necessary, then doing all the required abstractions.

literalt
minisat_convt::new_variable()
{
  literalt l;
  Minisat::Var tmp = solver.newVar();
  l.set(tmp, false);
  no_variables = tmp;
  return l;
}

void
minisat_convt::convert(const bvt &bv, Minisat::vec<Lit> &dest)
{
  dest.capacity(bv.size());

  for (unsigned int i = 0; i < bv.size(); i++) {
    if (!bv[i].is_false())
      dest.push(Minisat::mkLit(bv[i].var_no(), bv[i].sign()));
  }
  return;
}

literalt
minisat_convt::lnot(literalt a)
{
  a.invert();
  return a;
}

void
minisat_convt::set_equal(literalt a, literalt b)
{
  bvt bv;
  bv.resize(2);
  bv[0] = a;
  bv[1] = lnot(b);
  Minisat::vec<Lit> l;
  convert(bv, l);
  solver.addClause_(l);

  bv[0] = lnot(a);
  bv[1] = b;
  l.clear();
  convert(bv, l);
  solver.addClause_(l);
  return;
}

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
    smt_func_kind f, const smt_ast* const* _args, unsigned int numargs)
{
  const minisat_smt_ast *args[4];
  const minisat_smt_ast *result = NULL;
  unsigned int i;

  assert(numargs < 4 && "Too many arguments to minisat_convt::mk_func_app");
  for (i = 0; i < numargs; i++)
    args[i] = minisat_ast_downcast(_args[i]);

  switch (f) {
  default:
    std::cerr << "Unimplemented SMT function " << f << " in minisat convt"
              << std::endl;
    abort();
  }

  return result;
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
minisat_convt::mk_lit(const smt_ast *val)
{
  const minisat_smt_ast *a = minisat_ast_downcast(val);
  assert(a->sort->id == SMT_SORT_BOOL);
  assert(a->bv.size() == 1);

  literalt l = new_variable();
  set_equal(l, a->bv[0]);
  return l;
}

smt_ast*
minisat_convt::mk_smt_int(const mp_integer &intval __attribute__((unused)), bool sign __attribute__((unused)))
{
  std::cerr << "Can't create integers in minisat solver" << std::endl;
  abort();
}

smt_ast*
minisat_convt::mk_smt_real(const std::string &value __attribute__((unused)))
{
  std::cerr << "Can't create reals in minisat solver" << std::endl;
  abort();
}

smt_ast*
minisat_convt::mk_smt_bvint(const mp_integer &intval, bool sign,
                            unsigned int w)
{
  smt_sort *s = mk_sort(SMT_SORT_BV, w, sign);
  minisat_smt_ast *a = new minisat_smt_ast(s);
  a->bv.resize(w);
  int64_t u = intval.to_long();
  for (unsigned int i = 0; i < w; i++) {
    int64_t mask = (1 << i);
    bool val = u & mask;
    a->bv[i] = const_literal(val);
  }

  return a;
}

smt_ast*
minisat_convt::mk_smt_bool(bool boolval)
{
  literalt l = const_literal(boolval);

  smt_sort *s = mk_sort(SMT_SORT_BOOL);
  minisat_smt_ast *a = new minisat_smt_ast(s);
  a->bv.push_back(l);
  return a;
}

smt_ast*
minisat_convt::mk_smt_symbol(const std::string &name, const smt_sort *sort)
{
  // Like metasmt, minisat doesn't have a symbol table. So, build our own.
  symtable_type::const_iterator it = sym_table.find(name);
  if (it != sym_table.end())
    return it->second;

  // Otherwise, we need to build this AST ourselves.
  minisat_smt_ast *a = new minisat_smt_ast(sort);
  minisat_smt_sort *s = minisat_sort_downcast(sort);
  switch (sort->id) {
  case SMT_SORT_BOOL:
  {
    literalt l = new_variable();
    a->bv.push_back(l);
    break;
  }
  case SMT_SORT_BV:
  {
    // Bunch of fresh variables
    for (unsigned int i = 0; i < s->width; i++)
      a->bv.push_back(new_variable());
    break;
  }
  case SMT_SORT_ARRAY:
  {
    abort();
//    a = fresh_array(s, name);
    break;
  }
  default:
    std::cerr << "Unimplemented symbol type " << sort->id
              << " in minisat symbol creation" << std::endl;
    abort();
  }

  sym_table.insert(symtable_type::value_type(name, a));
  return a;
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
