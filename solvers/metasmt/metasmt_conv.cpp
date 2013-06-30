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
  : smt_convt(false, int_encoding, ns, is_cpp, false, true), ctx(), symbols(),
    astsyms(), sym_lookup(symbols, astsyms), bitblast_arrays(true)
{

  if (int_encoding) {
    std::cerr << "MetaSMT only supports QF_AUFBV logic, cannot make integer "
                 "mode solver" << std::endl;
    abort();
  }

  smt_post_init();
}

metasmt_convt::~metasmt_convt()
{
}

void
metasmt_convt::set_to(const expr2tc &expr, bool value)
{
  metasmt_smt_ast *val = mk_smt_bool(value);
  metasmt_smt_ast *expval = convert_ast(expr);

  predtags::equal_tag tag;
  result_type result = ctx(tag, val->restype, expval->restype);
  ctx.assertion(result);
}

prop_convt::resultt
metasmt_convt::dec_solve()
{
  bool res = ctx.solve();
  if (res) {
    return prop_convt::P_SATISFIABLE;
  } else {
    return prop_convt::P_UNSATISFIABLE;
  }
}

expr2tc
metasmt_convt::get(const expr2tc &expr)
{
  abort();
}

tvt
metasmt_convt::l_get(literalt a)
{
  // Right now, forget it, let's just go for outcome correctness before result
  // fetching.
  return tvt(tvt::TV_FALSE);
}

const std::string
metasmt_convt::solver_text()
{
  abort();
}


void
metasmt_convt::assert_lit(const literalt &l)
{
  std::stringstream ss;
  ss << "l" << l.var_no();
  metasmt_smt_ast *ast = sym_lookup(ss.str());
  ctx.assertion(ast->restype);
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
    break;
  }
  case SMT_FUNC_BVNEG:
  {
    bvtags::bvneg_tag tag;
    result = ctx(tag, args[0]->restype);
    break;
  }
  case SMT_FUNC_ITE:
  {
    predtags::ite_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype, args[2]->restype);
    break;
  }
  case SMT_FUNC_EQ:
  {
    // Confusion: there's a bvcomp_tag, which I thought was for comparing bvs,
    // but it turns out it's a comparison that results in a bv bit. Using
    // equal_tag appears to work for bvs and bools, and results in a bool.
    if (args[0]->sort->id == SMT_SORT_BOOL || args[0]->sort->id == SMT_SORT_BV){
      predtags::equal_tag tag;
      result = ctx(tag, args[0]->restype, args[1]->restype);
    } else {
      // Conceptually, array equalities shouldn't happen. But wait: symbol
      // assignments!
      if (args[0]->symname == "" && args[1]->symname == "") {
        std::cerr << "SMT equality not implemented in metasmt for sort "
                  << args[0]->sort->id << std::endl;
        abort();
      }

      // Instead of making an equality, store the rhs into the symbol table.
      // However we're not guarenteed that arg[1] is the rhs - so look for the
      // symbol. If both are symbols, fall back to args[0] being lhs.
      const metasmt_smt_ast *lhs = args[0];
      const metasmt_smt_ast *rhs = args[1];
      if (args[1]->symname != "") {
        lhs = args[1];
        rhs = args[0];
      }
      if (args[0]->symname != "") {
        lhs = args[0];
        rhs = args[1];
      }

      astsyms[lhs->symname] = rhs;
      // Return a true value, because that assignment is always true.
      metaSMT::logic::tag::true_tag tag;
      boost::any none;
      result = ctx(tag, none);
    }
    break;
  }
  case SMT_FUNC_NOTEQ:
  {
    predtags::nequal_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_SELECT:
  {
    arraytags::select_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_STORE:
  {
    arraytags::store_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype, args[2]->restype);
    break;
  }
  case SMT_FUNC_BVNOT:
  {
    bvtags::bvnot_tag tag;
    result = ctx(tag, args[0]->restype);
    break;
  }
  case SMT_FUNC_BVAND:
  {
    bvtags::bvand_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVNAND:
  {
    bvtags::bvnand_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVOR:
  {
    bvtags::bvor_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVXOR:
  {
    bvtags::bvxor_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVNOR:
  {
    bvtags::bvnor_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVNXOR:
  {
    bvtags::bvxnor_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSUB:
  {
    bvtags::bvsub_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVADD:
  {
    bvtags::bvadd_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVMUL:
  {
    bvtags::bvmul_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSLTE:
  {
    bvtags::bvsle_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSLT:
  {
    bvtags::bvslt_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSGTE:
  {
    bvtags::bvsge_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSGT:
  {
    bvtags::bvsgt_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVULTE:
  {
    bvtags::bvule_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVULT:
  {
    bvtags::bvult_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVUGTE:
  {
    bvtags::bvuge_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVUGT:
  {
    bvtags::bvugt_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_IMPLIES:
  {
    predtags::implies_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_XOR:
  {
    predtags::xor_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_AND:
  {
    predtags::and_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_OR:
  {
    predtags::or_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_CONCAT:
  {
    bvtags::concat_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVUDIV:
  {
    bvtags::bvudiv_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSDIV:
  {
    bvtags::bvsdiv_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVUMOD:
  {
    bvtags::bvurem_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSMOD:
  {
    bvtags::bvsrem_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVLSHR:
  {
    bvtags::bvshr_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVASHR:
  {
    bvtags::bvashr_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
  }
  case SMT_FUNC_BVSHL:
  {
    bvtags::bvshl_tag tag;
    result = ctx(tag, args[0]->restype, args[1]->restype);
    break;
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
    metasmt_smt_sort *dom = va_arg(ap, metasmt_smt_sort *); // Consider constness?
    metasmt_smt_sort *range = va_arg(ap, metasmt_smt_sort *);
    metasmt_smt_sort *s = new metasmt_smt_sort(k);
    assert(dom->id == SMT_SORT_BV && range->id == SMT_SORT_BV);
    s->arrdom_width = dom->width;
    s->arrrange_width = range->width;
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
  literalt l = new_variable();
  std::stringstream ss;
  ss << "l" << l.var_no();
  std::string str = ss.str();
  const metasmt_smt_ast *ma = metasmt_ast_downcast(mk_smt_symbol(str, s->sort));
  const metasmt_smt_ast *ms = metasmt_ast_downcast(s);

  predtags::equal_tag tag;
  result_type result = ctx(tag, ma->restype, ms->restype);
  ctx.assertion(result);
  return l;
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
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s, name);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
  }
  case SMT_SORT_ARRAY:
  {
    if (bitblast_arrays)
      return fresh_array(ms);
    metaSMT::logic::Array::tag::array_var_tag tag;
    tag.id = metaSMT::impl::new_var_id();
    tag.elem_width = ms->arrrange_width;
    tag.index_width = ms->arrdom_width;;
    boost::any none;
    result_type res = ctx(tag, none);
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s, name);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
  }
  case SMT_SORT_BOOL:
  {
    metaSMT::logic::tag::var_tag tag;
    tag.id = metaSMT::impl::new_var_id();
    boost::any none;
    result_type res = ctx(tag, none);
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s, name);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
  }
  default:
  {
    // Whatever; this is a struct/union that's being used by the tuple code to
    // represent a symbol or something. It'd not actually going to be used.
    // In the future, we can probably stop that happening.
    result_type res;
    return new metasmt_smt_ast(res, s, name);
  }
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
  bvtags::extract_tag tag;
  unsigned long foo = high; // "Types"
  unsigned long bar = low;
  const metasmt_smt_ast *ma = metasmt_ast_downcast(a);
  result_type res = ctx(tag, foo, bar, ma->restype);
  return new metasmt_smt_ast(res, s);
}

const metasmt_smt_ast *
metasmt_convt::fresh_array(const metasmt_smt_sort *ms)
{
  // No solver representation for this.
  unsigned long domain_width = ms->get_domain_width();
  unsigned long array_size = 1UL << domain_width;
  const smt_sort *range_sort = mk_sort(SMT_SORT_BV, ms->arrrange_width, false);

  metasmt_smt_ast *mast = new metasmt_smt_ast(ms);
  if (mast->is_unbounded_array())
    // Don't attempt to initialize.
    return mast;

  mast->array_fields.reserve(array_size);

  // Populate that array with a bunch of fresh bvs of the correct sort.
  unsigned long i;
  for (i = 0; i < array_size; i++) {
    const smt_ast *a = mk_fresh(range_sort, "metasmt_fresh_array::");
    mast->array_fields[i] = a;
  }

  return mast;
}

const smt_ast *
metasmt_convt::mk_select(const expr2tc &array, const expr2tc &idx,
                         const smt_sort *ressort)
{
  metasmt_smt_ast *ma = convert_ast(array);

  if (ma->is_unbounded_array())
    return mk_unbounded_select(ma, idx, ressort);

  assert(ma->array_fields.size() != 0);

  // If this is a constant index, simple. If not, not.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.constant_value.to_ulong();
    if (intval > ma->array_fields.size())
      // Return a fresh value.
      return mk_fresh(ressort, "metasmt_mk_select_badidx::");

    // Otherwise,
    return ma->array_fields[intval];
  }

  // What we have here is a nondeterministic index. Alas, compare with
  // everything.
  const smt_ast *fresh = mk_fresh(ressort, "metasmt_mk_select::");
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *args[2], *idxargs[2], *impargs[2], *accuml_props[2];
  unsigned long dom_width = ma->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  args[0] = fresh;
  idxargs[0] = real_idx;
  accuml_props[0] = mk_smt_bool(false);

  for (unsigned long i = 0; i < ma->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);
    args[1] = ma->array_fields[i];
    const smt_ast *val_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, args, 2);

    impargs[0] = idx_eq;
    impargs[1] = val_eq;

    accuml_props[1] = mk_func_app(bool_sort, SMT_FUNC_IMPLIES, impargs, 2);
    accuml_props[0] = mk_func_app(bool_sort, SMT_FUNC_OR, accuml_props, 2);
  }

  metasmt_smt_ast *a = metasmt_ast_downcast(accuml_props[0]);
  ctx.assertion(a->restype);

  return fresh;
}

const smt_ast *
metasmt_convt::mk_store(const expr2tc &array, const expr2tc &idx,
                        const expr2tc &value, const smt_sort *ressort)
{
  metasmt_smt_ast *ma = convert_ast(array);

  if (ma->is_unbounded_array())
    return mk_unbounded_store(ma, idx, value, ressort);

  assert(ma->array_fields.size() != 0);

  metasmt_smt_ast *mast = new metasmt_smt_ast(ressort, ma->array_fields);

  // If this is a constant index, simple. If not, not.
  if (is_constant_int2t(idx)) {
    const constant_int2t &intref = to_constant_int2t(idx);
    unsigned long intval = intref.constant_value.to_ulong();
    if (intval > ma->array_fields.size())
      return convert_ast(array);

    // Otherwise,
    mast->array_fields[intval] = convert_ast(value);
    return mast;
  }

  // Oh dear. We need to update /all the fields/ :(
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *real_value = convert_ast(value);
  const smt_ast *iteargs[2], *idxargs[2], *impargs[2], *accuml_props[2];
  unsigned long dom_width = mast->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  idxargs[0] = real_idx;
  iteargs[2] = real_value;

  for (unsigned long i = 0; i < mast->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);

    iteargs[0] = idx_eq;
    iteargs[1] = mast->array_fields[i];

    const smt_ast *new_val =
      mk_func_app(iteargs[1]->sort, SMT_FUNC_ITE, iteargs, 3);
    mast->array_fields[i] = new_val;
  }

  return mast;
}

const smt_ast *
metasmt_convt::mk_unbounded_select(const metasmt_smt_ast *ma,
                                   const expr2tc &idx,
                                   const smt_sort *ressort)
{

  // Heavily echoing mk_select,
  const smt_ast *fresh = mk_fresh(ressort, "metasmt_mk_unbounded_select::");
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *args[2], *idxargs[2], *impargs[2], *accuml_props[2];
  unsigned long dom_width = ma->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  args[0] = fresh;
  idxargs[0] = real_idx;
  accuml_props[0] = mk_smt_bool(false);

  for (metasmt_smt_ast::unbounded_list_type::const_iterator it =
       ma->array_values.begin(); it != ma->array_values.end(); it++) {
    idxargs[1] = it->first;
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);
    args[1] = it->second;
    const smt_ast *val_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, args, 2);

    impargs[0] = idx_eq;
    impargs[1] = val_eq;

    accuml_props[1] = mk_func_app(bool_sort, SMT_FUNC_IMPLIES, impargs, 2);
    accuml_props[0] = mk_func_app(bool_sort, SMT_FUNC_OR, accuml_props, 2);
  }

  metasmt_smt_ast *a = metasmt_ast_downcast(accuml_props[0]);
  ctx.assertion(a->restype);

  return fresh;
}

const smt_ast *
metasmt_convt::mk_unbounded_store(const metasmt_smt_ast *ma,
                                  const expr2tc &idx, const expr2tc &value,
                                  const smt_sort *ressort)
{
  // Actually super simple: we have no way of working out whether an older
  // assignment has expired (at least, not if there's any nondeterminism
  // anywyere), so don't bother, and just push this on the top.
  // We could optimise by looking at the topmost bunch of indexes and seeing
  // whether they match what we want, without any nondeterminism occurring,
  // but that's for another time.
  metasmt_smt_ast *mast = new metasmt_smt_ast(ressort, ma->array_values);
  const smt_ast *real_idx = convert_ast(idx);
  const smt_ast *real_val = convert_ast(value);

  mast->array_values.push_front(
      metasmt_smt_ast::unbounded_list_type::value_type(real_idx, real_val));
  return mast;
}
