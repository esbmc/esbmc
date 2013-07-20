#include <set>
#include <sstream>

#include <ansi-c/c_types.h>

#include "minisat_conv.h"

prop_convt *
create_new_minisat_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                          const optionst &options)
{
  return new minisat_convt(int_encoding, ns, is_cpp, options);
}

// Utility functions -- echoing CBMC quite a bit. The plan is to build up
// what's necessary, then doing all the required abstractions.

literalt
minisat_convt::new_variable()
{
  literalt l;
  Minisat::Var tmp = solver.newVar();
  l.set(tmp, false);
  no_variables = tmp+1;
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

literalt
minisat_convt::lselect(literalt a, literalt b, literalt c)
{  // a?b:c = (a AND b) OR (/a AND c)
  if(a==const_literal(true)) return b;
  if(a==const_literal(false)) return c;
  if(b==c) return b;

  bvt bv;
  bv.reserve(2);
  bv.push_back(land(a, b));
  bv.push_back(land(lnot(a), c));
  return bitblast_convt::lor(bv);
}

literalt
minisat_convt::lequal(literalt a, literalt b)
{
  return lnot(lxor(a, b));
}

literalt
minisat_convt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

literalt
minisat_convt::lxor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return lnot(b);
  if (b == const_literal(true)) return lnot(a);

  literalt output = new_variable();
  gate_xor(a, b, output);
  return output;
}

literalt
minisat_convt::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);

  literalt output = new_variable();
  gate_or(a, b, output);
  return output;
}

literalt
minisat_convt::land(literalt a, literalt b)
{
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;

  literalt output = new_variable();
  gate_and(a, b, output);
  return output;
}

void
minisat_convt::gate_xor(literalt a, literalt b, literalt o)
{
  // a xor b = o <==> (a' + b' + o')
  //                  (a + b + o' )
  //                  (a' + b + o)
  //                  (a + b' + o)
  bvt lits;

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(pos(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

void
minisat_convt::gate_or(literalt a, literalt b, literalt o)
{
  // a+b=c <==> (a' + c)( b' + c)(a + b + c')
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(a));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(pos(a));
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);
}

void
minisat_convt::gate_and(literalt a, literalt b, literalt o)
{
  // a*b=c <==> (a + o')( b + o')(a'+b'+o)
  bvt lits;

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(a));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(2);
  lits.push_back(pos(b));
  lits.push_back(neg(o));
  lcnf(lits);

  lits.clear();
  lits.reserve(3);
  lits.push_back(neg(a));
  lits.push_back(neg(b));
  lits.push_back(pos(o));
  lcnf(lits);
}

void
minisat_convt::setto(literalt a, bool val)
{
  bvt b;
  if (val)
    b.push_back(a);
  else
    b.push_back(lnot(a));

  Minisat::vec<Lit> l;
  convert(b, l);
  solver.addClause_(l);
  return;
}

void
minisat_convt::set_equal(literalt a, literalt b)
{
  if (a == const_literal(false)) {
    setto(b, false);
    return;
  } else if (b == const_literal(false)) {
    setto(a, false);
    return;
  } else if (a == const_literal(true)) {
    setto(b, true);
    return;
  } else if (b == const_literal(true)) {
    setto(a, true);
    return;
  }

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

void
minisat_convt::lcnf(const bvt &bv)
{
  bvt new_bv;

  if (process_clause(bv, new_bv))
    return;

  if (new_bv.empty())
    return;

  Minisat::vec<Lit> c;
  convert(bv, c);
  solver.addClause_(c);
  return;
}

minisat_convt::minisat_convt(bool int_encoding, const namespacet &_ns,
                             bool is_cpp, const optionst &_opts)
: smt_convt(true, int_encoding, _ns, is_cpp, false, true, true),
  array_convt(true, int_encoding, _ns, is_cpp, false),
  bitblast_convt(true, int_encoding, _ns, is_cpp, false, true, true),
  solver(), options(_opts)
{
  smt_post_init();
}

minisat_convt::~minisat_convt(void)
{
}

prop_convt::resultt
minisat_convt::dec_solve()
{
  add_array_constraints();
  bool res = solver.solve();
  if (res)
    return prop_convt::P_SATISFIABLE;
  else
    return prop_convt::P_UNSATISFIABLE;
}

void
minisat_convt::dump_bv(const bvt &bv) const
{
  for (unsigned int i = 0; i < bv.size(); i++) {
    if (bv[i] == const_literal(false))
      std::cerr << "0";
    else if (bv[i] == const_literal(true))
      std::cerr << "1";
    else
      std::cerr << "?";
  }

  std::cerr << " " << bv.size() << std::endl;
  return;
}

const std::string
minisat_convt::solver_text()
{
  return "MiniSAT";
}

tvt
minisat_convt::l_get(literalt l)
{
  if (l == const_literal(true))
    return tvt(tvt::TV_TRUE);
  else if (l == const_literal(false))
    return tvt(tvt::TV_FALSE);

  Minisat::lbool val = solver.modelValue(Minisat::mkLit(l.var_no(), l.sign()));
  int v = Minisat::toInt(val);
  if (v == 0)
    return tvt(tvt::TV_TRUE);
  else if (v == 1)
    return tvt(tvt::TV_FALSE);
  else
    return tvt(tvt::TV_UNKNOWN);
}

void
minisat_convt::assert_lit(const literalt &l)
{
  Minisat::vec<Lit> c;
  c.push(Minisat::mkLit(l.var_no(), l.sign()));
  solver.addClause_(c);
  return;
}

smt_ast*
minisat_convt::mk_func_app(const smt_sort *ressort __attribute__((unused)),
    smt_func_kind f, const smt_ast* const* _args, unsigned int numargs)
{
  const minisat_smt_ast *args[4];
  minisat_smt_ast *result = NULL;
  unsigned int i;

  assert(numargs < 4 && "Too many arguments to minisat_convt::mk_func_app");
  for (i = 0; i < numargs; i++)
    args[i] = minisat_ast_downcast(_args[i]);

  switch (f) {
  case SMT_FUNC_EQ:
  {
    assert(ressort->id == SMT_SORT_BOOL);
    return mk_ast_equality(args[0], args[1], ressort);
  }
  case SMT_FUNC_NOTEQ:
  {
    assert(ressort->id == SMT_SORT_BOOL);
    bitblast_smt_ast *a = mk_ast_equality(args[0], args[1], ressort);
    a->bv[0] = lnot(a->bv[0]);
    return a;
    break;
  }
  case SMT_FUNC_NOT:
  {
    literalt res = lnot(args[0]->bv[0]);
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(res);
    break;
  }
  case SMT_FUNC_OR:
  {
    literalt res = lor(args[0]->bv[0], args[1]->bv[0]);
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(res);
    break;
  }
  case SMT_FUNC_IMPLIES:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(limplies(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_ITE:
  {
    if (ressort->id == SMT_SORT_ARRAY) {
      result = array_ite(array_downcast(_args[0]), array_downcast(_args[1]),
                         array_downcast(_args[2]), ressort);
    } else {
      assert(args[1]->bv.size() == args[2]->bv.size());
      result = new minisat_smt_ast(ressort);
      for (unsigned int i = 0; i < args[1]->bv.size(); i++)
        result->bv.push_back(lselect(args[0]->bv[0], args[1]->bv[i],
                                     args[2]->bv[i]));
    }
    break;
  }
  case SMT_FUNC_AND:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(land(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_XOR:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lxor(args[0]->bv[0], args[1]->bv[0]));
    break;
  }
  case SMT_FUNC_BVADD:
  {
    literalt carry_in = const_literal(false);
    literalt carry_out;
    result = new minisat_smt_ast(ressort);
    full_adder(args[0]->bv, args[1]->bv, result->bv, carry_in, carry_out);
    break;
  }
  case SMT_FUNC_BVSUB:
  {
    literalt carry_in = const_literal(true);
    literalt carry_out;
    result = new minisat_smt_ast(ressort);
    bvt op1 = args[1]->bv;
    invert(op1);
    full_adder(args[0]->bv, op1, result->bv, carry_in, carry_out);
    break;
  }
  case SMT_FUNC_BVUGT:
  {
    // Same as LT flipped
    std::swap(args[0], args[1]);
    result = mk_func_app(ressort, SMT_FUNC_BVULT, args, 2);
    break;
  }
  case SMT_FUNC_BVUGTE:
  {
    // This is the negative of less-than
    result = mk_func_app(ressort, SMT_FUNC_BVULT, args, 2);
    result->bv[0].invert();
    break;
  }
  case SMT_FUNC_BVULT:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(unsigned_less_than(args[0]->bv, args[1]->bv));
    break;
  }
  case SMT_FUNC_BVULTE:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lt_or_le(true, args[0]->bv, args[1]->bv, false));
    break;
  }
  case SMT_FUNC_BVSGTE:
  {
    // This is the negative of less-than
    result = mk_func_app(ressort, SMT_FUNC_BVSLT, args, 2);
    result->bv[0].invert();
    break;
  }
  case SMT_FUNC_BVSLTE:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lt_or_le(true, args[0]->bv, args[1]->bv, true));
    break;
  }
  case SMT_FUNC_BVSGT:
  {
    // Same as LT flipped
    std::swap(args[0], args[1]);
    result = mk_func_app(ressort, SMT_FUNC_BVSLT, args, 2);
    break;
  }
  case SMT_FUNC_BVSLT:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.push_back(lt_or_le(false, args[0]->bv, args[1]->bv, true));
    break;
  }
  case SMT_FUNC_BVMUL:
  {
    result = new minisat_smt_ast(ressort);
    const minisat_smt_sort *sort0 = minisat_sort_downcast(args[0]->sort);
    const minisat_smt_sort *sort1 = minisat_sort_downcast(args[1]->sort);
    if (sort0->sign || sort1->sign) {
      signed_multiplier(args[0]->bv, args[1]->bv, result->bv);
    } else {
      unsigned_multiplier(args[0]->bv, args[1]->bv, result->bv);
    }
    break;
  }
  case SMT_FUNC_CONCAT:
  {
    result = new minisat_smt_ast(ressort);
    result->bv.insert(result->bv.begin(), args[0]->bv.begin(),
                      args[0]->bv.end());
    result->bv.insert(result->bv.begin(), args[1]->bv.begin(),
                      args[1]->bv.end());
    break;
  }
  case SMT_FUNC_BVAND:
  {
    result = new minisat_smt_ast(ressort);
    bvand(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVXOR:
  {
    result = new minisat_smt_ast(ressort);
    bvxor(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVOR:
  {
    result = new minisat_smt_ast(ressort);
    bvor(args[0]->bv, args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVNOT:
  {
    result = new minisat_smt_ast(ressort);
    bvnot(args[0]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVASHR:
  {
    result = new minisat_smt_ast(ressort);
    barrel_shift(args[0]->bv, bitblast_convt::shiftt::ARIGHT,
                 args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVLSHR:
  {
    result = new minisat_smt_ast(ressort);
    barrel_shift(args[0]->bv, bitblast_convt::shiftt::LRIGHT,
                 args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVSHL:
  {
    result = new minisat_smt_ast(ressort);
    barrel_shift(args[0]->bv, bitblast_convt::shiftt::LEFT,
                 args[1]->bv, result->bv);
    break;
  }
  case SMT_FUNC_BVSDIV:
  {
    bvt rem;
    result = new minisat_smt_ast(ressort);
    signed_divider(args[0]->bv, args[1]->bv, result->bv, rem);
    break;
  }
  case SMT_FUNC_BVUDIV:
  {
    bvt rem;
    result = new minisat_smt_ast(ressort);
    unsigned_divider(args[0]->bv, args[1]->bv, result->bv, rem);
    break;
  }
  case SMT_FUNC_BVSMOD:
  {
    bvt res;
    result = new minisat_smt_ast(ressort);
    signed_divider(args[0]->bv, args[1]->bv, res, result->bv);
    break;
  }
  case SMT_FUNC_BVUMOD:
  {
    bvt res;
    result = new minisat_smt_ast(ressort);
    unsigned_divider(args[0]->bv, args[1]->bv, res, result->bv);
    break;
  }
  case SMT_FUNC_BVNEG:
  {
    result = new minisat_smt_ast(ressort);
    negate(args[0]->bv, result->bv);
    break;
  }
  default:
    std::cerr << "Unimplemented SMT function \"" << smt_func_name_table[f]
              << "\" in minisat convt" << std::endl;
    abort();
  }

  return result;
}

const smt_ast *
minisat_convt::lit_to_ast(const literalt &l)
{
  const smt_sort *s = mk_sort(SMT_SORT_BOOL);
  minisat_smt_ast *a = new minisat_smt_ast(s);
  a->bv.push_back(l);
  return a;
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
    s = new minisat_smt_sort(k, uint, (bool)thebool);
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
    int64_t mask = (1ULL << i);
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
    return minisat_ast_downcast(it->second);

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
    a = fresh_array(s, name);
    break;
  }
  default:
  // Alas, tuple_fresh invokes us gratuitously with an invalid type. I can't
  // remember why, but it was justified at the time, for one solver, somewhere.
  // Either way, it should die in the future, but until then...
  return NULL;
#if 0
    std::cerr << "Unimplemented symbol type " << sort->id
              << " in minisat symbol creation" << std::endl;
    abort();
#endif
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
minisat_convt::mk_extract(const smt_ast *src, unsigned int high,
                          unsigned int low, const smt_sort *s)
{
  const minisat_smt_ast *mast = minisat_ast_downcast(src);
  minisat_smt_ast *result = new minisat_smt_ast(s);
  for (unsigned int i = low; i <= high; i++)
    result->bv.push_back(mast->bv[i]);

  return result;
}

void
minisat_convt::assign_array_symbol(const std::string &str, const smt_ast *a)
{
  sym_table[str] = a;
}
