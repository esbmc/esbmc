// Danger Will Robinson: this is not a C++ class, but in fact a template, and
// is included by bitblast_conv.h directly so that uses of it are instanciated
// correctly.

template <class subclass>
bitblast_convt<subclass>::bitblast_convt(bool enable_cache, bool int_encoding,
                               const namespacet &_ns, bool is_cpp,
                               bool tuple_support, bool bools_in_arrs,
                               bool can_init_inf_arrs)
  : subclass(enable_cache, int_encoding, _ns, is_cpp, tuple_support,
              bools_in_arrs, can_init_inf_arrs)
{
}

template <class subclass>
bitblast_convt<subclass>::~bitblast_convt()
{
}

template <class subclass>
void
bitblast_convt<subclass>::assert_ast(const smt_ast *a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  const bitblast_smt_ast *ba = bitblast_ast_downcast(a);
  assert_lit(ba->bv[0]);
  return;
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_func_app(const smt_sort *ressort,
                            smt_func_kind f, const smt_ast* const* _args,
                            unsigned int numargs)
{
  const bitblast_smt_ast *args[4];
  bitblast_smt_ast *result = NULL;
  unsigned int i;

  assert(numargs < 4 && "Too many arguments to bitblast_convt::mk_func_app");
  for (i = 0; i < numargs; i++)
    args[i] = bitblast_ast_downcast(_args[i]);

  switch (f) {
  case SMT_FUNC_EQ:
  {
    assert(ressort->id == SMT_SORT_BOOL);
    result = mk_ast_equality(args[0], args[1], ressort);
    break;
  }
  case SMT_FUNC_NOTEQ:
  {
    assert(ressort->id == SMT_SORT_BOOL);
    result = mk_ast_equality(args[0], args[1], ressort);
    result->bv[0] = this->lnot(result->bv[0]);
    break;
  }
  case SMT_FUNC_NOT:
  {
    literalt res = this->lnot(args[0]->bv[0]);
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(res);
    break;
  }
  case SMT_FUNC_OR:
  {
    literalt res = this->lor(args[0]->bv[0], args[1]->bv[0]);
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(res);
    break;
  }
  case SMT_FUNC_IMPLIES:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(this->limplies(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_ITE:
  {
    if (ressort->id == SMT_SORT_ARRAY) {
      return this->array_ite(_args[0], _args[1], _args[2], ressort);
    } else {
      assert(args[1]->bv.size() == args[2]->bv.size());
      result = new bitblast_smt_ast(ressort);
      for (unsigned int i = 0; i < args[1]->bv.size(); i++)
        result->bv.push_back(this->lselect(args[0]->bv[0], args[1]->bv[i],
                                     args[2]->bv[i]));
    }
    break;
  }
  case SMT_FUNC_AND:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(this->land(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_XOR:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(this->lxor(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_BVADD:
  {
    literalt carry_in = const_literal(false);
    literalt carry_out;
    result = new bitblast_smt_ast(ressort);
    full_adder(args[0]->bv, args[1]->bv, result->bv, carry_in, carry_out);
    break;
  }
  case SMT_FUNC_BVSUB:
  {
    literalt carry_in = const_literal(true);
    literalt carry_out;
    result = new bitblast_smt_ast(ressort);
    bvt op1 = args[1]->bv;
    invert(op1);
    full_adder(args[0]->bv, op1, result->bv, carry_in, carry_out);
    break;
  }
  case SMT_FUNC_BVUGT:
  {
    // Same as LT flipped
    const smt_ast *args2[2];
    args2[0] = _args[1];
    args2[1] = _args[0];
    return mk_func_app(ressort, SMT_FUNC_BVULT, args2, 2);
  }
  case SMT_FUNC_BVUGTE:
  {
    // This is the negative of less-than
    smt_ast *a = mk_func_app(ressort, SMT_FUNC_BVULT, _args, 2);
    a = mk_func_app(ressort, SMT_FUNC_NOT, &a, 1);
    return a;
  }
  case SMT_FUNC_BVULT:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(unsigned_less_than(args[0]->bv, args[1]->bv));
    break;
  }
  case SMT_FUNC_BVULTE:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(lt_or_le(true, args[0]->bv, args[1]->bv, false));
    break;
  }
  case SMT_FUNC_BVSGTE:
  {
    // This is the negative of less-than
    smt_ast *a = mk_func_app(ressort, SMT_FUNC_BVSLT, _args, 2);
    return mk_func_app(ressort, SMT_FUNC_NOT, &a, 1);
  }
  case SMT_FUNC_BVSLTE:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(lt_or_le(true, args[0]->bv, args[1]->bv, true));
    break;
  }
  case SMT_FUNC_BVSGT:
  {
    // Same as LT flipped
    const smt_ast *args2[2];
    args2[0] = _args[1];
    args2[1] = _args[0];
    return mk_func_app(ressort, SMT_FUNC_BVSLT, args2, 2);
  }
  case SMT_FUNC_BVSLT:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.push_back(lt_or_le(false, args[0]->bv, args[1]->bv, true));
    break;
  }
  case SMT_FUNC_BVMUL:
  {
    result = new bitblast_smt_ast(ressort);
    const bitblast_smt_sort *sort0 = bitblast_sort_downcast(args[0]->sort);
    const bitblast_smt_sort *sort1 = bitblast_sort_downcast(args[1]->sort);
    if (sort0->sign || sort1->sign) {
      signed_multiplier(args[0]->bv, args[1]->bv, result->bv);
    } else {
      unsigned_multiplier(args[0]->bv, args[1]->bv, result->bv);
    }
    break;
  }
  case SMT_FUNC_CONCAT:
  {
    result = new bitblast_smt_ast(ressort);
    result->bv.insert(result->bv.begin(), args[0]->bv.begin(),
                      args[0]->bv.end());
    result->bv.insert(result->bv.begin(), args[1]->bv.begin(),
                      args[1]->bv.end());
    break;
  }
  case SMT_FUNC_BVAND:
  {
    result = new bitblast_smt_ast(ressort);
    bvand(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVXOR:
  {
    result = new bitblast_smt_ast(ressort);
    bvxor(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVOR:
  {
    result = new bitblast_smt_ast(ressort);
    bvor(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVNOT:
  {
    result = new bitblast_smt_ast(ressort);
    bvnot(args[0]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVASHR:
  {
    result = new bitblast_smt_ast(ressort);
    barrel_shift(args[0]->bv, bitblast_convt::shiftt::ARIGHT,
                 args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVLSHR:
  {
    result = new bitblast_smt_ast(ressort);
    barrel_shift(args[0]->bv, bitblast_convt::shiftt::LRIGHT,
                 args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVSHL:
  {
    result = new bitblast_smt_ast(ressort);
    barrel_shift(args[0]->bv, bitblast_convt::shiftt::LEFT,
                 args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVSDIV:
  {
    bvt rem;
    result = new bitblast_smt_ast(ressort);
    signed_divider(args[0]->bv, args[1]->bv, result->bv, rem);
    break;
  }
  case SMT_FUNC_BVUDIV:
  {
    bvt rem;
    result = new bitblast_smt_ast(ressort);
    unsigned_divider(args[0]->bv, args[1]->bv, result->bv, rem);
    break;
  }
  case SMT_FUNC_BVSMOD:
  {
    bvt res;
    result = new bitblast_smt_ast(ressort);
    signed_divider(args[0]->bv, args[1]->bv, res, result->bv);
    break;
  }
  case SMT_FUNC_BVUMOD:
  {
    bvt res;
    result = new bitblast_smt_ast(ressort);
    unsigned_divider(args[0]->bv, args[1]->bv, res, result->bv);
    break;
  }
  case SMT_FUNC_BVNEG:
  {
    result = new bitblast_smt_ast(ressort);
    negate(args[0]->bv, result->bv);
    break;
  }
  default:
    std::cerr << "Unimplemented SMT function \"" << this->smt_func_name_table[f]
              << "\" in bitblast convt" << std::endl;
    abort();
  }

  return result;
}

template <class subclass>
smt_sort*
bitblast_convt<subclass>::mk_sort(smt_sort_kind k, ...)
{
  va_list ap;
  bitblast_smt_sort *s = NULL, *dom, *range;
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
    s = new bitblast_smt_sort(k, uint, (bool)thebool);
    break;
  case SMT_SORT_ARRAY:
    dom = va_arg(ap, bitblast_smt_sort *); // Consider constness?
    range = va_arg(ap, bitblast_smt_sort *);
    s = new bitblast_smt_sort(k, dom->width, range->width);
    break;
  case SMT_SORT_BOOL:
    s = new bitblast_smt_sort(k);
    break;
  default:
    std::cerr << "Unimplemented SMT sort " << k << " in Minisat conversion"
              << std::endl;
    abort();
  }

  return s;
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_smt_int(const mp_integer &intval __attribute__((unused)), bool sign __attribute__((unused)))
{
  std::cerr << "Can't create integers in bitblast solver" << std::endl;
  abort();
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_smt_real(const std::string &value __attribute__((unused)))
{
  std::cerr << "Can't create reals in bitblast solver" << std::endl;
  abort();
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_smt_bvint(const mp_integer &intval, bool sign,
                            unsigned int w)
{
  smt_sort *s = mk_sort(SMT_SORT_BV, w, sign);
  bitblast_smt_ast *a = new bitblast_smt_ast(s);
  a->bv.resize(w);
  int64_t u = intval.to_long();
  for (unsigned int i = 0; i < w; i++) {
    int64_t mask = (1ULL << i);
    bool val = u & mask;
    a->bv[i] = const_literal(val);
  }

  return a;
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_smt_bool(bool boolval)
{
  literalt l = const_literal(boolval);

  smt_sort *s = mk_sort(SMT_SORT_BOOL);
  bitblast_smt_ast *a = new bitblast_smt_ast(s);
  a->bv.push_back(l);
  return a;
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_smt_symbol(const std::string &name, const smt_sort *sort)
{
  // Like metasmt, bitblast doesn't have a symbol table. So, build our own.
  symtable_type::iterator it = sym_table.find(name);
  if (it != sym_table.end())
    return it->second;

  // Otherwise, we need to build this AST ourselves.
  bitblast_smt_ast *a = new bitblast_smt_ast(sort);
  smt_ast *result = a;
  const bitblast_smt_sort *s = bitblast_sort_downcast(sort);
  switch (sort->id) {
  case SMT_SORT_BOOL:
  {
    literalt l = this->new_variable();
    a->bv.push_back(l);
    break;
  }
  case SMT_SORT_BV:
  {
    // Bunch of fresh variables
    for (unsigned int i = 0; i < s->width; i++)
      a->bv.push_back(this->new_variable());
    break;
  }
  case SMT_SORT_ARRAY:
  {
    delete a;
    result = this->fresh_array(s, name);
    break;
  }
  default:
  // Alas, tuple_fresh invokes us gratuitously with an invalid type. I can't
  // remember why, but it was justified at the time, for one solver, somewhere.
  // Either way, it should die in the future, but until then...
  return NULL;
#if 0
    std::cerr << "Unimplemented symbol type " << sort->id
              << " in bitblast symbol creation" << std::endl;
    abort();
#endif
  }

  sym_table.insert(symtable_type::value_type(name, result));
  return result;
}

template <class subclass>
smt_sort*
bitblast_convt<subclass>::mk_struct_sort(const type2tc &t __attribute__((unused)))
{
    abort();
}

template <class subclass>
smt_sort*
bitblast_convt<subclass>::mk_union_sort(const type2tc &t __attribute__((unused)))
{
    abort();
}

template <class subclass>
smt_ast*
bitblast_convt<subclass>::mk_extract(const smt_ast *src, unsigned int high,
                          unsigned int low, const smt_sort *s)
{
  const bitblast_smt_ast *mast = bitblast_ast_downcast(src);
  bitblast_smt_ast *result = new bitblast_smt_ast(s);
  for (unsigned int i = low; i <= high; i++)
    result->bv.push_back(mast->bv[i]);

  return result;
}

template <class subclass>
bitblast_smt_ast *
bitblast_convt<subclass>::mk_ast_equality(const smt_ast *_a,
                                const smt_ast *_b,
                                const smt_sort *ressort)
{
  const bitblast_smt_ast *a = bitblast_ast_downcast(_a);
  const bitblast_smt_ast *b = bitblast_ast_downcast(_b);

  switch (a->sort->id) {
  case SMT_SORT_BOOL:
  {
    literalt res = this->lequal(a->bv[0], b->bv[0]);
    bitblast_smt_ast *n = new bitblast_smt_ast(a->sort);
    n->bv.push_back(res);
    return n;
  }
  case SMT_SORT_BV:
  {
    bitblast_smt_ast *n = new bitblast_smt_ast(ressort);
    n->bv.push_back(equal(a->bv, b->bv));
    return n;
  }
  case SMT_SORT_ARRAY:
  {
    // XXX - so, if we have different array encodings in the future then this
    // might get quite funky. Leave it until then, though.
    std::cerr << "No direct array equality support in bitblast converter"
              << std::endl;
    abort();
  }
  default:
    std::cerr << "Invalid sort " << a->sort->id << " for equality in bitblast"
              << std::endl;
    abort();
  }
}

template <class subclass>
expr2tc
bitblast_convt<subclass>::get_bool(const smt_ast *a)
{
  tvt t = this->l_get(a);
  if (t.is_true())
    return true_expr;
  else if (t.is_false())
    return false_expr;
  else
    return expr2tc();
}


template <class subclass>
tvt
bitblast_convt<subclass>::l_get(const smt_ast *a)
{
  const bitblast_smt_ast *mast = bitblast_ast_downcast(a);
  return this->l_get(mast->bv[0]);
}

template <class subclass>
expr2tc
bitblast_convt<subclass>::get_bv(const type2tc &t, const smt_ast *a)
{
  const bitblast_smt_ast *mast = bitblast_ast_downcast(a);
  unsigned int sz = t->get_width();
  assert(sz <= 64 && "Your integers are larger than a uint64_t");
  assert(mast->bv.size() == sz);
  uint64_t accuml = 0;
  for (unsigned int i = 0; i < sz; i++) {
    uint64_t mask = 1 << i;
    tvt t = this->l_get(mast->bv[i]);
    if (t.is_true()) {
      accuml |= mask;
    } else if (t.is_false()) {
      ; // It's zero
    } else {
      ; // It's undefined in this model. So may as well be zero.
    }
  }

  return constant_int2tc(t, BigInt(accuml));
}

template <class subclass>
const smt_ast *
bitblast_convt<subclass>::make_disjunct(const smt_convt::ast_vec &v)
{
  bvt bv;
  bv.reserve(v.size());
  for (smt_convt::ast_vec::const_iterator it = v.begin(); it != v.end(); it++)
    bv.push_back(bitblast_ast_downcast(*it)->bv[0]);

  literalt l = lor(bv);

  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  bitblast_smt_ast *ba = new bitblast_smt_ast(boolsort);
  ba->bv.push_back(l);
  return ba;
}

template <class subclass>
const smt_ast *
bitblast_convt<subclass>::make_conjunct(const smt_convt::ast_vec &v)
{
  bvt bv;
  bv.reserve(v.size());
  for (smt_convt::ast_vec::const_iterator it = v.begin(); it != v.end(); it++)
    bv.push_back(bitblast_ast_downcast(*it)->bv[0]);

  literalt l = land(bv);

  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  bitblast_smt_ast *ba = new bitblast_smt_ast(boolsort);
  ba->bv.push_back(l);
  return ba;
}

// ******************************  Bitblast foo *******************************

template <class subclass>
bool
bitblast_convt<subclass>::process_clause(const bvt &bv, bvt &dest)
{

  dest.clear();

  // empty clause! this is UNSAT
  if (bv.empty()) return false;

  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin();
       it != bv.end();
       it++)
  {
    literalt l = *it;

    if (l.is_true())
      return true;  // clause satisfied

    if (l.is_false())
      continue;

    // prevent duplicate literals
    if (s.insert(l).second)
      dest.push_back(l);

    if (s.find(this->lnot(l)) != s.end())
      return true;  // clause satisfied
  }

  return false;
}

template <class subclass>
void
bitblast_convt<subclass>::eliminate_duplicates(const bvt &bv, bvt &dest)
{
  std::set<literalt> s;

  dest.reserve(bv.size());

  for(bvt::const_iterator it=bv.begin(); it!=bv.end(); it++)
  {
    if(s.insert(*it).second)
      dest.push_back(*it);
  }
}

template <class subclass>
void
bitblast_convt<subclass>::full_adder(const bvt &op0, const bvt &op1, bvt &output,
                          literalt carry_in, literalt &carry_out)
{
  assert(op0.size() == op1.size());
  output.reserve(op0.size());

  carry_out = carry_in;

  for (unsigned int i = 0; i < op0.size(); i++) {
    output.push_back(this->lxor(this->lxor(op0[i], op1[i]), carry_out));
    carry_out = carry(op0[i], op1[i], carry_out);
  }

  return;
}

template <class subclass>
literalt
bitblast_convt<subclass>::carry(literalt a, literalt b, literalt c)
{
  bvt tmp;
  tmp.reserve(3);
  tmp.push_back(this->land(a, b));
  tmp.push_back(this->land(a, c));
  tmp.push_back(this->land(b, c));
  return lor(tmp);
}

template <class subclass>
literalt
bitblast_convt<subclass>::unsigned_less_than(const bvt &arg0, const bvt &arg1)
{
  bvt tmp = arg1;
  invert(tmp);
  return this->lnot(carry_out(arg0, tmp, const_literal(true)));
}

template <class subclass>
void
bitblast_convt<subclass>::unsigned_multiplier(const bvt &op0, const bvt &op1, bvt &output)
{
  output.resize(op0.size());

  for (unsigned int i = 0; i < op0.size(); i++)
    output[i] = const_literal(false);

  for (unsigned int i = 0; i < op0.size(); i++) {
    if (op0[i] != const_literal(false)) {
      bvt tmpop;
      tmpop.reserve(op0.size());

      for (unsigned int idx = 0; idx < i; idx++)
        tmpop.push_back(const_literal(false));

      for (unsigned int idx = i; idx < op0.size(); idx++)
        tmpop.push_back(this->land(op1[idx-i], op0[i]));

      bvt tmpadd;
      literalt dummy;
      full_adder(output, tmpop, tmpadd, const_literal(false), dummy);
      output = tmpadd;
    }
  }
}

template <class subclass>
void
bitblast_convt<subclass>::signed_multiplier(const bvt &op0, const bvt &op1, bvt &output)
{
  assert(op0.size() == op1.size() && op0.size() != 0);
  literalt sign0 = op0[op0.size()-1];
  literalt sign1 = op1[op1.size()-1];

  bvt neg0, neg1;
  cond_negate(op0, neg0, sign0);
  cond_negate(op1, neg1, sign1);

  bvt tmp;
  unsigned_multiplier(neg0, neg1, tmp);

  literalt res_sign = this->lxor(sign0, sign1);

  cond_negate(tmp, output, res_sign);
}

template <class subclass>
void
bitblast_convt<subclass>::cond_negate(const bvt &vals, bvt &out, literalt cond)
{
  bvt inv;
  negate(vals, inv);

  out.resize(vals.size());

  for (unsigned int i = 0; i < vals.size(); i++)
    out[i] = this->lselect(cond, inv[i], vals[i]);
  
  return;
}

template <class subclass>
void
bitblast_convt<subclass>::negate(const bvt &inp, bvt &oup)
{
  oup.resize(inp.size());
  bvt inv = inp;
  invert(inv);

  literalt dummy;
  incrementer(inv, const_literal(true), dummy, oup);
  return;
}

template <class subclass>
void
bitblast_convt<subclass>::incrementer(const bvt &inp, const literalt &carryin,
                           literalt carryout, bvt &oup)
{
  carryout = carryin;

  for (unsigned int i = 0; i < inp.size(); i++) {
    literalt new_carry = this->land(carryout, inp[i]);
    oup[i] = this->lxor(inp[i], carryout);
    carryout = new_carry;
  }

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::signed_divider(const bvt &op0, const bvt &op1, bvt &res,
                              bvt &rem)
{
  assert(op0.size() == op1.size());

  bvt _op0;
  bvt _op1;
  _op0.resize(op0.size());
  _op1.resize(op1.size());

  literalt sign0 = op0[op0.size() - 1];
  literalt sign1 = op1[op1.size() - 1];

  bvt neg0;
  bvt neg1;
  negate(op0, neg0);
  negate(op1, neg1);

  for (unsigned int i = 0; i < op0.size(); i++)
    _op0[i] = this->lselect(sign0, neg0[i], op0[i]);

  for (unsigned int i = 0; i < op1.size(); i++)
    _op1[i] = this->lselect(sign1, neg1[i], op1[i]);

  unsigned_divider(_op0, _op1, res, rem);

  bvt neg_res;
  bvt neg_rem;
  negate(res, neg_res);
  negate(rem, neg_rem);

  literalt result_sign = this->lxor(sign0, sign1);

  for (unsigned int i = 0; i < res.size(); i++)
    res[i] = this->lselect(result_sign, neg_res[i], res[i]);

  for (unsigned int i = 0; i < rem.size(); i++)
    rem[i] = this->lselect(result_sign, neg_rem[i], rem[i]);

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::unsigned_divider(const bvt &op0, const bvt &op1, bvt &res,
                                bvt &rem)
{
  assert(op0.size() == op1.size());
  unsigned int width = op0.size();
  res.resize(width);
  rem.resize(width);

  literalt is_not_zero = lor(op1);

  for (unsigned int i = 0; i < width; i++) {
    res[i] = this->new_variable();
    rem[i] = this->new_variable();
  }

  bvt product;
  unsigned_multiplier_no_overflow(res, op1, product);

  // "res*op1 + rem = op0"

  bvt sum;
  adder_no_overflow(product, rem, sum);

  literalt is_equal = equal(sum, op0);

  this->assert_lit(this->limplies(is_not_zero, is_equal));

  // "op1 != 0 => rem < op1"

  this->assert_lit(this->limplies(is_not_zero, lt_or_le(false, rem, op1, false)));

  // "op1 != 0 => res <= op0"

  this->assert_lit(this->limplies(is_not_zero, lt_or_le(true, rem, op0, false)));
}

template <class subclass>
void
bitblast_convt<subclass>::unsigned_multiplier_no_overflow(const bvt &op0, const bvt &op1,
                                               bvt &res)
{
  assert(op0.size() == op1.size());
  bvt _op0 = op0, _op1 = op1;

  if (is_constant(_op1))
    std::swap(_op0, _op1);

  res.resize(op0.size());

  for (unsigned int i = 0; i < res.size(); i++)
    res[i] = const_literal(false);

  for (unsigned int sum = 0; sum < op0.size(); sum++) {
    if (op0[sum] != const_literal(false)) {
      bvt tmpop;

      tmpop.reserve(res.size());

      for (unsigned int idx = 0; idx < sum; idx++)
        tmpop.push_back(const_literal(false));

      for (unsigned int idx = sum; idx < res.size(); idx++)
        tmpop.push_back(this->land(op1[idx-sum], op0[sum]));

      bvt copy = res;
      adder_no_overflow(copy, tmpop, res);

      for (unsigned int idx = op1.size() - sum; idx < op1.size(); idx++) {
        literalt tmp = this->land(op1[idx], op0[sum]);
        tmp.invert();
        this->assert_lit(tmp);
      }
    }
  }
}

template <class subclass>
void
bitblast_convt<subclass>::adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res,
                                 bool subtract, bool is_signed)
{
  assert(op0.size() == op1.size());
  unsigned int width = op0.size();;
  bvt tmp_op1 = op1;
  if (subtract)
    negate(op1, tmp_op1);

  if (is_signed) {
    literalt old_sign = op0[width-1];
    literalt sign_the_same = this->lequal(op0[width-1], tmp_op1[width-1]);
    literalt carry;
    full_adder(op0, tmp_op1, res, const_literal(subtract), carry);
    literalt stop_overflow =
      this->land(sign_the_same, this->lxor(op0[width-1], old_sign));
    stop_overflow.invert();
    this->assert_lit(stop_overflow);
  } else {
    literalt carry_out;
    full_adder(op0, tmp_op1, res, const_literal(subtract), carry_out);
    if (subtract) {
      this->assert_lit(carry_out);
    } else {
      carry_out.invert();
      this->assert_lit(carry_out);
    }
  }

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::adder_no_overflow(const bvt &op0, const bvt &op1, bvt &res)
{
  res.resize(op0.size());

  literalt carry_out = const_literal(false);
  for (unsigned int i = 0; i < op0.size(); i++) {
    literalt op0_bit = op0[i];

    res[i] = this->lxor(this->lxor(op0_bit, op1[i]), carry_out);
    carry_out = carry(op0_bit, op1[i], carry_out);
  }

  carry_out.invert();
  this->assert_lit(carry_out);
}

template <class subclass>
bool
bitblast_convt<subclass>::is_constant(const bvt &bv)
{
  for (unsigned int i = 0; i < bv.size(); i++)
    if (!bv[i].is_constant())
      return false;
  return true;
}

template <class subclass>
literalt
bitblast_convt<subclass>::carry_out(const bvt &a, const bvt &b, literalt c)
{
  literalt carry_out = c;

  for (unsigned int i = 0; i < a.size(); i++)
    carry_out = carry(a[i], b[i], carry_out);

  return carry_out;
}

template <class subclass>
literalt
bitblast_convt<subclass>::equal(const bvt &op0, const bvt &op1)
{
  assert(op0.size() == op1.size());
  bvt tmp;
  tmp.reserve(op0.size());

  for (unsigned int i = 0; i < op0.size(); i++)
    tmp.push_back(this->lequal(op0[i], op1[i]));

  literalt res = land(tmp);
  return res;
}

template <class subclass>
literalt
bitblast_convt<subclass>::lt_or_le(bool or_equal, const bvt &bv0, const bvt &bv1,
                        bool is_signed)
{
  assert(bv0.size() == bv1.size());
  literalt top0 = bv0[bv0.size() - 1], top1 = bv1[bv1.size() - 1];

  bvt inv_op1 = bv1;
  invert(inv_op1);
  literalt carry = carry_out(bv0, inv_op1, const_literal(true));

  literalt result;
  if (is_signed)
    result = this->lxor(this->lequal(top0, top1), carry);
  else
    result = this->lnot(carry);

  if (or_equal)
    result = this->lor(result, equal(bv0, bv1));

  return result;
}

template <class subclass>
void
bitblast_convt<subclass>::invert(bvt &bv)
{
  for (unsigned int i = 0; i < bv.size(); i++)
    bv[i] = this->lnot(bv[i]);
}

template <class subclass>
void
bitblast_convt<subclass>::barrel_shift(const bvt &op, const shiftt s, const bvt &dist,
                            bvt &out)
{
  unsigned long d = 1;
  out = op;

  // FIXME: clip this on oversized shifts?
  for (unsigned int pos = 0; pos < dist.size(); pos++) {
    if (dist[pos] != const_literal(false)) {
      bvt tmp;
      shift(out, s, d, tmp);

      for (unsigned int i = 0; i < op.size(); i++)
        out[i] = this->lselect(dist[pos], tmp[i], out[i]);
    }

    d <<= 1;
  }
}

template <class subclass>
void
bitblast_convt<subclass>::shift(const bvt &inp, const shiftt &s, unsigned long d, bvt &out)
{
  out.resize(inp.size());

  for (unsigned int i = 0; i < inp.size(); i++) {
    literalt l;

    switch (s) {
    case LEFT:
      l = ((d <= i) ? inp[i-d] : const_literal(false));
      break;
    case ARIGHT:
      l = ((i+d < inp.size()) ? inp[i+d] : inp[inp.size()-1]);
      break;
    case LRIGHT:
      l = ((i+d < inp.size()) ? inp[i+d] : const_literal(false));
      break;
    }

    out[i] = l;
  }

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::bvand(const bvt &bv0, const bvt &bv1, bvt &output)
{
  assert(bv0.size() == bv1.size());
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(this->land(bv0[i], bv1[i]));

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::bvor(const bvt &bv0, const bvt &bv1, bvt &output)
{
  assert(bv0.size() == bv1.size());
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(this->lor(bv0[i], bv1[i]));

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::bvxor(const bvt &bv0, const bvt &bv1, bvt &output)
{
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(this->lxor(bv0[i], bv1[i]));

  return;
}

template <class subclass>
void
bitblast_convt<subclass>::bvnot(const bvt &bv0, bvt &output)
{
  output.clear();
  output.reserve(bv0.size());

  for (unsigned int i = 0; i < bv0.size(); i++)
    output.push_back(this->lnot(bv0[i]));

  return;
}

template <class subclass>
literalt
bitblast_convt<subclass>::land(const bvt &bv)
{
  if (bv.size() == 0)
    return const_literal(true);
  else if (bv.size() == 1)
    return bv[0];
  else if (bv.size() == 2)
    return this->land(bv[0], bv[1]);

  unsigned int trues = 0;
  for (unsigned int i = 0; i < bv.size(); i++) {
    if (bv[i] == const_literal(false))
      return const_literal(false);
    else if (bv[i] == const_literal(true))
      trues++;
  }

  if (trues == bv.size())
    return const_literal(true);

  bvt new_bv;

  eliminate_duplicates(bv, new_bv);

  literalt lit = this->new_variable();

  for (unsigned int i = 0; i < new_bv.size(); i++) {
    bvt lits;
    lits.reserve(2);
    lits.push_back(pos(new_bv[i]));
    lits.push_back(neg(lit));
    this->lcnf(lits);
  }

  bvt lits;
  lits.reserve(new_bv.size() + 1);

  for (unsigned int i = 0; i < new_bv.size(); i++)
    lits.push_back(neg(new_bv[i]));

  lits.push_back(pos(lit));
  this->lcnf(lits);

  return lit;
}

template <class subclass>
literalt
bitblast_convt<subclass>::lor(const bvt &bv)
{
  if (bv.size() == 0) return const_literal(false);
  else if (bv.size() == 1) return bv[0];
  else if (bv.size() == 2) return this->lor(bv[0], bv[1]);

  for (unsigned int i = 0; i < bv.size(); i++)
    if (bv[i] == const_literal(true))
      return const_literal(true);

  bvt new_bv;
  eliminate_duplicates(bv, new_bv);

  literalt literal = this->new_variable();
  for (unsigned int i = 0; i < new_bv.size(); i++) {
    bvt lits;
    lits.reserve(2);
    lits.push_back(neg(new_bv[i]));
    lits.push_back(pos(literal));
    this->lcnf(lits);
  }

  bvt lits;
  lits.reserve(new_bv.size() + 1);

  for (unsigned int i = 0; i < new_bv.size(); i++)
    lits.push_back(pos(new_bv[i]));

  lits.push_back(neg(literal));
  this->lcnf(lits);

  return literal;
}
