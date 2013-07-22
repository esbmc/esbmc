#include <string>
#include <sstream>

#include "metasmt_conv.h"

#include <solvers/smt/smt_conv.h>

metasmt_convt::metasmt_convt(bool int_encoding, bool is_cpp,
                             const namespacet &ns)
  : smt_convt(false, int_encoding, ns, is_cpp, false, true, true),
    ctx(), symbols(), astsyms(), sym_lookup(symbols, astsyms),
    array_indexes(), array_values(), array_updates()
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

smt_convt::resultt
metasmt_convt::dec_solve()
{
#ifdef SOLVER_BITBLAST_ARRAYS
  add_array_constraints();
#endif

  bool res = ctx.solve();
  if (res) {
    return smt_convt::P_SATISFIABLE;
  } else {
    return smt_convt::P_UNSATISFIABLE;
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

tvt
metasmt_convt::l_get(const smt_ast *a)
{
  // Right now, forget it, let's just go for outcome correctness before result
  // fetching.
  return tvt(tvt::TV_FALSE);
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
#ifdef SOLVER_BITBLAST_ARRAYS
    if (s->id == SMT_SORT_ARRAY)
      return array_ite(args[0], metasmt_array_downcast(_args[1]),
                       metasmt_array_downcast(_args[2]),
                       metasmt_sort_downcast(s));
#endif

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
      if (args[0]->sort->id != SMT_SORT_ARRAY ||
          args[1]->sort->id != SMT_SORT_ARRAY) {
        std::cerr << "SMT equality not implemented in metasmt for sort "
                  << args[0]->sort->id << std::endl;
        abort();
      }

      const metasmt_array_ast *side1, *side2;
      side1 = metasmt_array_downcast(_args[0]);
      side2 = metasmt_array_downcast(_args[1]);

      if (side1->symname == "" && side2->symname == "") {
        std::cerr << "Malformed MetaSMT array equality" << std::endl;
        abort();
      }

      // Instead of making an equality, store the rhs into the symbol table.
      // However we're not guarenteed that arg[1] is the rhs - so look for the
      // symbol. If both are symbols, fall back to args[0] being lhs.
      const metasmt_array_ast *lhs = side1;
      const metasmt_array_ast *rhs = side2;
      if (side2->symname != "") {
        lhs = side2;
        rhs = side1;
      }
      if (side1->symname != "") {
        lhs = side1;
        rhs = side2;
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
    metasmt_smt_sort *s =
      new metasmt_smt_sort(k, range->data_width, dom->data_width);
    assert(dom->id == SMT_SORT_BV && range->id == SMT_SORT_BV);
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
    ::metaSMT::logic::QF_BV::tag::bvsint_tag lolwat;
    boost::tuple<long, unsigned long> bees(theint.to_long(), w);
    boost::any face(bees);
    r = ctx(lolwat, face);
  } else {
    ::metaSMT::logic::QF_BV::tag::bvuint_tag lolwat;
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
#ifdef SOLVER_BITBLAST_ARRAYS
    return fresh_array(ms, name);
#else
    metaSMT::logic::Array::tag::array_var_tag tag;
    tag.id = metaSMT::impl::new_var_id();
    tag.elem_width = ms->arrrange_width;
    tag.index_width = ms->arrdom_width;;
    boost::any none;
    result_type res = ctx(tag, none);
    metasmt_smt_ast *mast = new metasmt_smt_ast(res, s, name);
    sym_lookup.insert(mast, tag.id, name);
    return mast;
#endif
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

#ifdef SOLVER_BITBLAST_ARRAYS
const smt_ast *
metasmt_convt::fresh_array(const metasmt_smt_sort *ms, const std::string &name)
{
  // No solver representation for this.
  unsigned long domain_width = ms->get_domain_width();
  unsigned long array_size = 1UL << domain_width;
  const smt_sort *range_sort = mk_sort(SMT_SORT_BV, ms->arrrange_width, false);

  metasmt_array_ast *mast = new metasmt_array_ast(ms);
  mast->symname = name;
  sym_lookup.insert(mast, 0, name); // yolo
  if (mast->is_unbounded_array()) {
    // Don't attempt to initialize. Store the fact that we've allocated a
    // fresh new array.
    mast->base_array_id = array_indexes.size();
    mast->array_update_num = 0;
    std::set<expr2tc> tmp_set;
    array_indexes.push_back(tmp_set);

    std::vector<std::list<struct array_select> > tmp2;
    array_values.push_back(tmp2);

    std::list<struct array_select> tmp25;
    array_values[mast->base_array_id].push_back(tmp25);

    std::vector<struct array_with> tmp3;
    array_updates.push_back(tmp3);

    array_subtypes.push_back(ms->arrrange_width);

    return mast;
  }

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
  metasmt_array_ast *ma = metasmt_array_downcast(convert_ast(array));

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
  metasmt_array_ast *ma = metasmt_array_downcast(convert_ast(array));

  if (ma->is_unbounded_array())
    return mk_unbounded_store(ma, idx, convert_ast(value), ressort);

  assert(ma->array_fields.size() != 0);

  metasmt_array_ast *mast =
    new metasmt_array_ast(ressort, ma->array_fields);

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
metasmt_convt::mk_unbounded_select(const metasmt_array_ast *ma,
                                   const expr2tc &real_idx,
                                   const smt_sort *ressort)
{
  // Record that we've accessed this index.
  array_indexes[ma->base_array_id].insert(real_idx);

  // Generate a new free variable
  smt_ast *a = mk_fresh(ressort, "mk_unbounded_select");

  struct array_select sel;
  sel.src_array_update_num = ma->array_update_num;
  sel.idx = real_idx;
  sel.val = a;
  // Record this index
  array_values[ma->base_array_id][ma->array_update_num].push_back(sel);

  return a;
}

const smt_ast *
metasmt_convt::mk_unbounded_store(const metasmt_array_ast *ma,
                                  const expr2tc &idx, const smt_ast *value,
                                  const smt_sort *ressort)
{
  // Record that we've accessed this index.
  array_indexes[ma->base_array_id].insert(idx);

  // More nuanced: allocate a new array representation.
  metasmt_array_ast *newarr = new metasmt_array_ast(ressort);
  newarr->base_array_id = ma->base_array_id;
  newarr->array_update_num = array_updates[ma->base_array_id].size();

  // Record update
  struct array_with w;
  w.is_ite = false;
  w.idx = idx;
  w.u.w.src_array_update_num = ma->array_update_num;
  w.u.w.val = value;
  array_updates[ma->base_array_id].push_back(w);

  // Also file a new select record for this point in time.
  std::list<struct array_select> tmp;
  array_values[ma->base_array_id].push_back(tmp);

  // Result is the new array id goo.
  return newarr;
}

const metasmt_array_ast *
metasmt_convt::array_ite(const metasmt_smt_ast *cond,
                         const metasmt_array_ast *true_arr,
                         const metasmt_array_ast *false_arr,
                         const metasmt_smt_sort *thesort)
{

  if (true_arr->is_unbounded_array())
    return unbounded_array_ite(cond, true_arr, false_arr, thesort);

  // For each element, make an ite.
  assert(true_arr->array_fields.size() != 0 &&
         true_arr->array_fields.size() == false_arr->array_fields.size());
  metasmt_array_ast *mast = new metasmt_array_ast(thesort);
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

const metasmt_array_ast *
metasmt_convt::unbounded_array_ite(const metasmt_smt_ast *cond,
                                   const metasmt_array_ast *true_arr,
                                   const metasmt_array_ast *false_arr,
                                   const metasmt_smt_sort *thesort)
{
  // Precondition for a lot of goo: that the two arrays are the same, at
  // different points in time.
  assert(true_arr->base_array_id == false_arr->base_array_id &&
         "ITE between two arrays with different bases are unsupported");

  metasmt_array_ast *newarr = new metasmt_array_ast(thesort);
  newarr->base_array_id = true_arr->base_array_id;
  newarr->array_update_num = array_updates[true_arr->base_array_id].size();

  struct array_with w;
  w.is_ite = true;
  w.idx = expr2tc();
  w.u.i.src_array_update_true = true_arr->array_update_num;
  w.u.i.src_array_update_false = false_arr->array_update_num;
  w.u.i.cond = cond;
  array_updates[true_arr->base_array_id].push_back(w);

  // Also file a new select record for this point in time.
  std::list<struct array_select> tmp;
  array_values[true_arr->base_array_id].push_back(tmp);

  return newarr;
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

  metasmt_array_ast *mast = new metasmt_array_ast(arr_sort);

  const smt_ast *init = convert_ast(init_val);
  if (!int_encoding && is_bool_type(init_val) && no_bools_in_arrays)
    init = make_bool_bit(init);

  if (arr_sort->is_unbounded_array()) {
    delete mast;
    mast = metasmt_array_downcast(fresh_array(arr_sort, "array_of_unbounded"));
    array_of_vals.insert(std::pair<unsigned, const smt_ast *>
                                  (mast->base_array_id, init));
  } else {
    unsigned long array_size = 1UL << domain_width;
    for (unsigned long i = 0; i < array_size; i++)
      mast->array_fields.push_back(init);
  }

  return mast;
}

void
metasmt_convt::add_array_constraints(void)
{

  for (unsigned int i = 0; i < array_indexes.size(); i++) {
    add_array_constraints(i);
  }

  return;
}

void
metasmt_convt::add_array_constraints(unsigned int arr)
{
  // Right: we need to tie things up regarding these bitvectors. We have a
  // set of indexes...
  const std::set<expr2tc> &indexes = array_indexes[arr];

  // What we're going to build is a two-dimensional vector ish of each element
  // at each point in time. Expensive, but meh.
  std::vector<std::vector<const smt_ast *> > real_array_values;

  // Subtype is thus
  const smt_sort *subtype = mk_sort(SMT_SORT_BV, array_subtypes[arr], false);

  // Pre-allocate all the storage.
  real_array_values.resize(array_values[arr].size());
  for (unsigned int i = 0; i < real_array_values.size(); i++)
    real_array_values[i].resize(indexes.size());

  // Compute a mapping between indexes and an element in the vector. These
  // are ordered by how std::set orders them, not by history or anything. Or
  // even the element index.
  std::map<expr2tc, unsigned> idx_map;
  for (std::set<expr2tc>::const_iterator it = indexes.begin();
       it != indexes.end(); it++)
    idx_map.insert(std::pair<expr2tc, unsigned>(*it, idx_map.size()));

  // Initialize the first set of elements.
  std::map<unsigned, const smt_ast*>::const_iterator it =
    array_of_vals.find(arr);
  if (it != array_of_vals.end()) {
    collate_array_values(real_array_values[0], idx_map, array_values[arr][0],
        subtype, it->second);
  } else {
    collate_array_values(real_array_values[0], idx_map, array_values[arr][0],
        subtype);
  }

  add_initial_ackerman_constraints(real_array_values[0], idx_map, subtype);

  // Now repeatedly execute transitions between states.
  for (unsigned int i = 0; i < real_array_values.size() - 1; i++)
    execute_array_trans(real_array_values, arr, i, idx_map, subtype);

}

void
metasmt_convt::execute_array_trans(
                            std::vector<std::vector<const smt_ast *> > &data,
                                   unsigned int arr,
                                   unsigned int idx,
                                   const std::map<expr2tc, unsigned> &idx_map,
                                   const smt_sort *subtype)
{
  // Steps: First, fill the destination vector with either free variables, or
  // the free variables that resulted for selects corresponding to that item.
  // Then apply update or ITE constraints.
  // Then apply equalities between the old and new values.

  std::vector<const smt_ast *> &dest_data = data[idx+1];
  collate_array_values(dest_data, idx_map, array_values[arr][idx+1], subtype);

  // Two updates that could have occurred for this array: a simple with, or
  // an ite.
  const array_with &w = array_updates[arr][idx];
  if (w.is_ite) {
    // Turn every index element into an ITE representing this operation. Every
    // single element is addressed and updated; no further constraints are
    // needed. Not even the ackerman ones, in fact, because instances of that
    // from previous array updates will feed through to here (speculation).

    unsigned int true_idx = w.u.i.src_array_update_true;
    unsigned int false_idx = w.u.i.src_array_update_false;
    assert(true_idx < idx + 1 && false_idx < idx + 1);
    const std::vector<const smt_ast *> &true_vals = data[true_idx];
    const std::vector<const smt_ast *> &false_vals = data[false_idx];
    const smt_ast *cond = w.u.i.cond;

    // Each index value becomes an ITE between each source value.
    const smt_ast *args[3];
    args[0] = cond;
    for (unsigned int i = 0; i < idx_map.size(); i++) {
      args[1] = true_vals[i];
      args[2] = false_vals[i];
      dest_data[i] = mk_func_app(subtype, SMT_FUNC_ITE, args, 3);
    }
  } else {
    // Place a constraint on the updated variable; add equality constraints
    // between the older version and this version. And finally add ackerman
    // constraints.

    // So, the updated element,
    std::map<expr2tc, unsigned>::const_iterator it = idx_map.find(w.idx);
    assert(it != idx_map.end());

    const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
    const expr2tc &update_idx_expr = it->first;
    const smt_ast *update_idx_ast = convert_ast(update_idx_expr);
    unsigned int updated_idx = it->second;
    const smt_ast *updated_value = w.u.w.val;

    // Assign in its value.
    dest_data[updated_idx] = updated_value;

    // Now look at all those other indexes...
    assert(w.u.w.src_array_update_num < idx+1);
    const std::vector<const smt_ast *> &source_data =
      data[w.u.w.src_array_update_num];

    const smt_ast *args[3];
    unsigned int i = 0;
    for (std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.begin();
         it2 != idx_map.end(); it2++, i++) {
      if (it2->second == updated_idx)
        continue;

      // Generate an ITE. If the index is nondeterministically equal to the
      // current index, take the updated value, otherwise the original value.
      // This departs from the CBMC implementation, in that they explicitly
      // use implies and ackerman constraints.
      // FIXME: benchmark the two approaches. For now, this is shorter.
      args[0] = update_idx_ast;
      args[1] = convert_ast(it2->first);
      args[0] = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      args[1] = updated_value;
      args[2] = source_data[i];
      dest_data[i] = mk_func_app(subtype, SMT_FUNC_ITE, args, 3);
    }
  }
}

void
metasmt_convt::collate_array_values(std::vector<const smt_ast *> &vals,
                                    const std::map<expr2tc, unsigned> &idx_map,
                                    const std::list<struct array_select> &idxs,
                                    const smt_sort *subtype,
                                    const smt_ast *init_val)
{
  // So, the value vector should be allocated but not initialized,
  assert(vals.size() == idx_map.size());

  // First, make everything null,
  for (std::vector<const smt_ast *>::iterator it = vals.begin();
       it != vals.end(); it++)
    *it = NULL;

  // Now assign in all free variables created as a result of selects.
  for (std::list<struct array_select>::const_iterator it = idxs.begin();
       it != idxs.end(); it++) {
    std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.find(it->idx);
    assert(it2 != idx_map.end());
    vals[it2->second] = it->val;
  }

  // Initialize everything else to either a free variable or the initial value.
  if (init_val == NULL) {
    // Free variables, except where free variables tied to selects have occurred
    for (std::vector<const smt_ast *>::iterator it = vals.begin();
         it != vals.end(); it++) {
      if (*it == NULL)
        *it = mk_fresh(subtype, "collate_array_vals::");
    }
  } else {
    // We need to assign the initial value in, except where there's already
    // a select/index, in which case we assert that the values are equal.
    const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
    for (std::vector<const smt_ast *>::iterator it = vals.begin();
         it != vals.end(); it++) {
      if (*it == NULL) {
        *it = init_val;
      } else {
        const smt_ast *args[2];
        args[0] = *it;
        args[1] = init_val;
        assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }
  }

  // Fin.
}

void
metasmt_convt::add_initial_ackerman_constraints(
                                  const std::vector<const smt_ast *> &vals,
                                  const std::map<expr2tc,unsigned> &idx_map,
                                  const smt_sort *subtype)
{
  // Lolquadratic,
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  for (std::map<expr2tc, unsigned>::const_iterator it = idx_map.begin();
       it != idx_map.end(); it++) {
    const smt_ast *outer_idx = convert_ast(it->first);
    for (std::map<expr2tc, unsigned>::const_iterator it2 = idx_map.begin();
         it2 != idx_map.end(); it2++) {
      const smt_ast *inner_idx = convert_ast(it2->first);

      // If they're the same idx, they're the same value.
      const smt_ast *args[2];
      args[0] = outer_idx;
      args[1] = inner_idx;
      const smt_ast *idxeq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

      args[0] = vals[it->second];
      args[1] = vals[it2->second];
      const smt_ast *valeq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

      args[0] = idxeq;
      args[1] = valeq;
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_IMPLIES, args, 2)));
    }
  }
}

#endif /* SOLVER_BITBLAST_ARRAYS */
