#include "metasmt_conv.h"

#include <solvers/prop/prop_conv.h>

// To avoid having to build metaSMT into multiple files,
prop_convt *
create_new_metasmt_solver(bool int_encoding, bool is_cpp, const namespacet &ns)
{
  return new metasmt_convt(int_encoding, is_cpp, ns);
}

metasmt_convt::metasmt_convt(bool int_encoding, bool is_cpp,
                             const namespacet &ns)
  : smt_convt(false, int_encoding, ns, is_cpp, false), ctx(), symbols(),
    astsyms(), sym_lookup(symbols, astsyms)
{

  smt_post_init();
}

metasmt_convt::~metasmt_convt()
{
}

void
metasmt_convt::set_to(const expr2tc &expr, bool value)
{
  abort();
}

prop_convt::resultt
metasmt_convt::dec_solve()
{
  abort();
}

expr2tc
metasmt_convt::get(const expr2tc &expr)
{
  abort();
}

tvt
metasmt_convt::l_get(literalt a)
{
  abort();
}

const std::string
metasmt_convt::solver_text()
{
  abort();
}


void
metasmt_convt::assert_lit(const literalt &l)
{
  abort();
}

smt_ast *
metasmt_convt::mk_func_app(const smt_sort *s, smt_func_kind k,
                           const smt_ast **_args, unsigned int numargs)
{
  const metasmt_smt_ast *args[4];
  result_type result;
  unsigned int i;

  assert(numargs < 4 && "Too many arguments to metasmt_convt::mk_func_app");
  for (i = 0; i < numargs; i++)
    args[i] = metasmt_ast_downcast(_args[i]);

  switch (k) {
  case SMT_FUNC_NOT:
  {
    predtags::not_tag tag;
    result = ctx(tag, args[0]->restype);
  }
  default:
    std::cerr << "Unsupported SMT function " << k << " in metasmt conv"
              << std::endl;
    abort();
  }

  return new metasmt_smt_ast(result, s);
}

smt_sort *
metasmt_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_BV:
  {
    unsigned long u = va_arg(ap, unsigned long);
    return new metasmt_smt_sort(k, u);
  }
  case SMT_SORT_ARRAY:
  {
    unsigned long d = va_arg(ap, unsigned long);
    unsigned long r = va_arg(ap, unsigned long);
    metasmt_smt_sort *s = new metasmt_smt_sort(k);
    s->arrdom_width = d;
    s->arrrange_width = r;
    return s;
  }
  case SMT_SORT_BOOL:
    return new metasmt_smt_sort(k);
  case SMT_SORT_INT:
  case SMT_SORT_REAL:
  default:
    std::cerr << "SMT SORT type " << k << " is unsupported by metasmt"
              << std::endl;
  }
}

literalt
metasmt_convt::mk_lit(const smt_ast *s)
{
  abort();
}

smt_ast *
metasmt_convt::mk_smt_int(const mp_integer &theint, bool sign)
{
  abort();
}

smt_ast *
metasmt_convt::mk_smt_real(const std::string &str)
{
  abort();
}

smt_ast *
metasmt_convt::mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w)
{

  const smt_sort *s = mk_sort(SMT_SORT_BV, w, sign);

  result_type r;
  if (sign) {
    metaSMT::solver::bvtags::bvsint_tag lolwat;
    boost::tuple<long, unsigned long> bees(theint.to_long(), w);
    boost::any face(bees);
    r = ctx(lolwat, face);
  } else {
    metaSMT::solver::bvtags::bvuint_tag lolwat;
    boost::tuple<unsigned long, unsigned long> bees(theint.to_long(), w);
    boost::any face(bees);
    r = ctx(lolwat, face);
  }

  return new metasmt_smt_ast(r, s);
}

smt_ast *
metasmt_convt::mk_smt_bool(bool val)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  result_type r;
  if (val) {
    metaSMT::logic::tag::true_tag tag;
    boost::any none;
    r = ctx(tag, none);
  } else {
    metaSMT::logic::tag::false_tag tag;
    boost::any none;
    r = ctx(tag, none);
  }

  return new metasmt_smt_ast(r, s);
}

smt_ast *
metasmt_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  // It seems metaSMT makes us generate our own symbol table. Which is annoying.
  // More annoying is that it seems to be a one way process, in which some kind
  // of AST node id is stored, and you can't convert it back to the array, bv
  // or predicate that it originally was.
  // So, keep a secondary symbol table.

  // First: is this already in the symbol table?
  smt_ast *ast = sym_lookup(name);
  if (ast != NULL)
    // Yes, return that piece of AST. This means that, as an opposite to what
    // Z3 does, we get the free variable associated with the name.
    return ast;

  // Nope; generate an appropriate ast.
  const metasmt_smt_sort *ms = metasmt_sort_downcast(s);
  switch (s->id) {
  case SMT_SORT_BV:
  {
    metaSMT::logic::QF_BV::tag::var_tag tag;
    tag.id = metaSMT::impl::new_var_id();
    tag.width = ms->width;
    boost::any none;
    result_type res = ctx(tag, none);
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
  }
  case SMT_SORT_ARRAY:
  {
    metaSMT::logic::Array::tag::array_var_tag tag;
    tag.id = metaSMT::impl::new_var_id();
    tag.elem_width = ms->arrrange_width;
    tag.index_width = ms->arrdom_width;;
    boost::any none;
    result_type res = ctx(tag, none);
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
  }
  case SMT_SORT_BOOL:
  {
    metaSMT::logic::tag::var_tag tag;
    tag.id = metaSMT::impl::new_var_id();
    boost::any none;
    result_type res = ctx(tag, none);
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
  }
  default:
    std::cerr << "Unrecognized smt sort in metasmt mk_symbol" << std::endl;
    abort();
  }
}

smt_sort *
metasmt_convt::mk_struct_sort(const type2tc &type)
{
  abort();
}

smt_sort *
metasmt_convt::mk_union_sort(const type2tc &type)
{
  abort();
}

smt_ast *
metasmt_convt::mk_extract(const smt_ast *a, unsigned int high,
                          unsigned int low, const smt_sort *s)
{
  abort();
}
