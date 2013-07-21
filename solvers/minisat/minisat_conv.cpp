#include <set>
#include <sstream>

#include <ansi-c/c_types.h>

#include "minisat_conv.h"

smt_convt *
create_new_minisat_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                          const optionst &options)
{
  return new minisat_convt(int_encoding, ns, is_cpp, options);
}

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

void
minisat_convt::setto(literalt a, bool val)
{
  bvt b;
  if (val)
    b.push_back(a);
  else
    b.push_back(cnf_convt::lnot(a));

  Minisat::vec<Lit> l;
  convert(b, l);
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
: cnf_convt(true, int_encoding, _ns, is_cpp, false, true, true),
  solver(), options(_opts)
{
  smt_post_init();
}

minisat_convt::~minisat_convt(void)
{
}

smt_convt::resultt
minisat_convt::dec_solve()
{
  add_array_constraints();
  bool res = solver.solve();
  if (res)
    return smt_convt::P_SATISFIABLE;
  else
    return smt_convt::P_UNSATISFIABLE;
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

tvt
minisat_convt::l_get(const smt_ast *a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  const bitblast_smt_ast *ba = bitblast_ast_downcast(a);
  return l_get(ba->bv[0]);
}

void
minisat_convt::assert_lit(const literalt &l)
{
  Minisat::vec<Lit> c;
  c.push(Minisat::mkLit(l.var_no(), l.sign()));
  solver.addClause_(c);
  return;
}

void
minisat_convt::assign_array_symbol(const std::string &str, const smt_ast *a)
{
  sym_table[str] = a;
}
