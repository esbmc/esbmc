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
  : smt_convt(false, int_encoding, ns, is_cpp, false, true, true),
    ctx(), symbols(), astsyms(), sym_lookup(symbols, astsyms),
    bitblast_arrays(true)
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
                           const smt_ast * const *_args, unsigned int numargs)
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
    if (s->id == SMT_SORT_ARRAY)
      return array_ite(args[0], args[1], args [2], metasmt_sort_downcast(s));

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
      return fresh_array(ms, name);
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
metasmt_convt::fresh_array(const metasmt_smt_sort *ms, const std::string &name)
{
  // No solver representation for this.
  unsigned long domain_width = ms->get_domain_width();
  unsigned long array_size = 1UL << domain_width;
  const smt_sort *range_sort = mk_sort(SMT_SORT_BV, ms->arrrange_width, false);

  metasmt_smt_ast *mast = new metasmt_smt_ast(ms);
  mast->symname = name;
  sym_lookup.insert(mast, 0, name); // yolo
  if (mast->is_unbounded_array())
    // Don't attempt to initialize.
    return mast;

  mast->array_fields.reserve(array_size);

  // Populate that array with a bunch of fresh bvs of the correct sort.
  unsigned long i;
  for (i = 0; i < array_size; i++) {
    const smt_ast *a = mk_fresh(range_sort, "metasmt_fresh_array::");
    mast->array_fields.push_back(a);
  }

  return mast;
}

const smt_ast *
metasmt_convt::mk_select(const expr2tc &array, const expr2tc &idx,
                         const smt_sort *ressort)
{
  assert(ressort->id != SMT_SORT_ARRAY);
  metasmt_smt_ast *ma = convert_ast(array);

  if (ma->is_unbounded_array())
    return mk_unbounded_select(ma, convert_ast(idx), ressort);

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
  const smt_ast *args[2], *idxargs[2], *impargs[2];
  unsigned long dom_width = ma->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  args[0] = fresh;
  idxargs[0] = real_idx;

  for (unsigned long i = 0; i < ma->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);
    args[1] = ma->array_fields[i];
    const smt_ast *val_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, args, 2);

    impargs[0] = idx_eq;
    impargs[1] = val_eq;

    const smt_ast *res = mk_func_app(bool_sort, SMT_FUNC_IMPLIES, impargs, 2);
    ctx.assertion(metasmt_ast_downcast(res)->restype);
  }

  return fresh;
}

const smt_ast *
metasmt_convt::mk_store(const expr2tc &array, const expr2tc &idx,
                        const expr2tc &value, const smt_sort *ressort)
{
  metasmt_smt_ast *ma = convert_ast(array);

  if (ma->is_unbounded_array())
    return mk_unbounded_store(ma, convert_ast(idx), convert_ast(value),
                              ressort);

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
  const smt_ast *iteargs[3], *idxargs[2], *impargs[2], *accuml_props[2];
  unsigned long dom_width = mast->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  idxargs[0] = real_idx;
  iteargs[1] = real_value;

  for (unsigned long i = 0; i < mast->array_fields.size(); i++) {
    idxargs[1] = mk_smt_bvint(BigInt(i), false, dom_width);
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);

    iteargs[0] = idx_eq;
    iteargs[2] = mast->array_fields[i];

    const smt_ast *new_val =
      mk_func_app(iteargs[1]->sort, SMT_FUNC_ITE, iteargs, 3);
    mast->array_fields[i] = new_val;
  }

  return mast;
}

const smt_ast *
metasmt_convt::mk_unbounded_select(const metasmt_smt_ast *ma,
                                   const metasmt_smt_ast *real_idx,
                                   const smt_sort *ressort)
{

  // Heavily echoing mk_select,
  bool unmatched_free = false;
  const smt_ast *the_default = ma->default_unbounded_val;
  if (the_default == NULL) {
    the_default = mk_fresh(ressort, "metasmt_mk_unbounded_select::");
    unmatched_free = true;
  }

  const smt_ast *idxargs[2], *impargs[2], *iteargs[3];
  unsigned long dom_width = ma->sort->get_domain_width();
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  idxargs[0] = real_idx;
  iteargs[2] = the_default;

  // Make one gigantic ITE, from the back upwards.
  for (metasmt_smt_ast::unbounded_list_type::const_reverse_iterator it =
       ma->array_values.rbegin(); it != ma->array_values.rend(); it++) {
    idxargs[1] = it->first;
    const smt_ast *idx_eq = mk_func_app(bool_sort, SMT_FUNC_EQ, idxargs, 2);
    iteargs[0] = idx_eq;
    iteargs[1] = it->second;

    iteargs[2] = mk_func_app(ressort, SMT_FUNC_ITE, iteargs, 3);
  }

  // If there's no default value, and we selected something out, we have to
  // ensure that future reads of the same position get the same value. So we
  // have to store this free value :(. Do that by storing this select on top.
  metasmt_smt_ast *mast = const_cast<metasmt_smt_ast *>(ma); // yolo
  mast->array_values.push_front(
      metasmt_smt_ast::unbounded_list_type::value_type(real_idx, iteargs[2]));

  return iteargs[2];
}

const smt_ast *
metasmt_convt::mk_unbounded_store(const metasmt_smt_ast *ma,
                                  const smt_ast *idx, const smt_ast *value,
                                  const smt_sort *ressort)
{
  // Actually super simple: we have no way of working out whether an older
  // assignment has expired (at least, not if there's any nondeterminism
  // anywyere), so don't bother, and just push this on the top.
  // We could optimise by looking at the topmost bunch of indexes and seeing
  // whether they match what we want, without any nondeterminism occurring,
  // but that's for another time.
  metasmt_smt_ast *mast = new metasmt_smt_ast(ressort, ma->array_values,
                                              ma->default_unbounded_val);

  mast->array_values.push_front(
      metasmt_smt_ast::unbounded_list_type::value_type(idx, value));
  return mast;
}

const metasmt_smt_ast *
metasmt_convt::array_ite(const metasmt_smt_ast *cond,
                         const metasmt_smt_ast *true_arr,
                         const metasmt_smt_ast *false_arr,
                         const metasmt_smt_sort *thesort)
{

  if (true_arr->is_unbounded_array())
    return unbounded_array_ite(cond, true_arr, false_arr, thesort);

  // For each element, make an ite.
  assert(true_arr->array_fields.size() != 0 &&
         true_arr->array_fields.size() == false_arr->array_fields.size());
  metasmt_smt_ast *mast = new metasmt_smt_ast(thesort);
  const smt_ast *args[3];
  args[0] = cond;
  unsigned long i;
  for (i = 0; i < true_arr->array_fields.size(); i++) {
    // One ite pls.
    args[1] = true_arr->array_fields[i];
    args[2] = false_arr->array_fields[i];
    const smt_ast *res = mk_func_app(args[1]->sort, SMT_FUNC_ITE, args, 3);
    mast->array_fields.push_back(metasmt_ast_downcast(res));
  }

  return mast;
}

const metasmt_smt_ast *
metasmt_convt::make_conditional_unbounded_ite_join(const metasmt_smt_ast *base,
            const smt_ast *condition,
            metasmt_smt_ast::unbounded_list_type::const_reverse_iterator it,
            metasmt_smt_ast::unbounded_list_type::const_reverse_iterator end)
{
  const smt_ast *args[3];

  // Iterate through, making selects and stores.
  for (; it != end; it++) {
    args[0] = condition;
    args[1] = it->second;
    args[2] = mk_unbounded_select(base, it->first, it->second->sort);
    const smt_ast *newval =
      mk_func_app(it->second->sort, SMT_FUNC_ITE, args, 3);
    const metasmt_smt_ast *new_base =
      mk_unbounded_store(base, it->first, newval, base->sort);

    // Save memory etc.
    delete base;
    base = new_base;
  }

  return base;
}

const metasmt_smt_ast *
metasmt_convt::unbounded_array_ite(const metasmt_smt_ast *cond,
                                   const metasmt_smt_ast *true_arr,
                                   const metasmt_smt_ast *false_arr,
                                   const metasmt_smt_sort *thesort)
{
  // Ok, we have two lists of values. They're _guarenteed_ to be from the
  // same array (array ite's can only occur in phis). So, discover the common
  // tail of assignments that the lists posess. Then make conditional stores
  // of the remaining values into a new array.
  metasmt_smt_ast::unbounded_list_type common_tail;

  metasmt_smt_ast::unbounded_list_type::const_reverse_iterator it, it2;
  it = true_arr->array_values.rbegin();
  it2 = false_arr->array_values.rbegin();
  for (; it != true_arr->array_values.rend(); it++, it2++) {
    if (it->first == it2->first) {
      common_tail.push_back(*it);
    } else {
      break;
    }
  }

  // Generate the base array.
  metasmt_smt_ast *base = new metasmt_smt_ast(true_arr->sort, common_tail,
                                              true_arr->default_unbounded_val);

  // Iterate over each set of other values, selecting the old and updated
  // element, then generating an ITE based on the input condition.
  base = make_conditional_unbounded_ite_join(base, cond, it,
                                             true_arr->array_values.rend());
  const smt_ast *inverted_cond =
    mk_func_app(cond->sort, SMT_FUNC_NOT, &cond, 1);
  base = make_conditional_unbounded_ite_join(base, inverted_cond, it2,
                                             false_arr->array_values.rend());

  return base;
}

const smt_ast *
metasmt_convt::convert_array_of(const expr2tc &init_val,
                                unsigned long domain_width)
{
  const smt_sort *dom_sort = mk_sort(SMT_SORT_BV, domain_width, false);
  const smt_sort *idx_sort = convert_sort(init_val->type);

  if (!int_encoding && is_bool_type(init_val) && no_bools_in_arrays)
    idx_sort = mk_sort(SMT_SORT_BV, 1, false);

  const metasmt_smt_sort *arr_sort =
    metasmt_sort_downcast(mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort));

  metasmt_smt_ast *mast = new metasmt_smt_ast(arr_sort);

  const smt_ast *init = convert_ast(init_val);
  if (!int_encoding && is_bool_type(init_val) && no_bools_in_arrays)
    init = make_bool_bit(init);

  if (arr_sort->is_unbounded_array()) {
    mast->default_unbounded_val = init;
  } else {
    unsigned long array_size = 1UL << domain_width;
    for (unsigned long i = 0; i < array_size; i++)
      mast->array_fields.push_back(init);
  }

  return mast;
}
