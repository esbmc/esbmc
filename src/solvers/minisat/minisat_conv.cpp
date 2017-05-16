#include <set>
#include <sstream>
#include <util/c_types.h>
#include <minisat_conv.h>

smt_convt *
create_new_minisat_solver(bool int_encoding, const namespacet &ns,
                          const optionst &options,
                          tuple_iface **tuple_api __attribute__((unused)),
                          array_iface **array_api __attribute__((unused)))
{
  minisat_convt *conv = new minisat_convt(int_encoding, ns,options);
  return conv;
}

literalt
minisat_convt::new_variable()
{
  literalt l;
  Minisat::Var tmp = solver.newVar();
  l.set(tmp, false);
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
                             const optionst &_opts)
: cnf_iface(),
  cnf_convt(static_cast<cnf_iface*>(this)),
  bitblast_convt(int_encoding, _ns, static_cast<sat_iface*>(this)),
  solver(), options(_opts), false_asserted(false)
{
}

minisat_convt::~minisat_convt(void)
{
}

smt_convt::resultt
minisat_convt::dec_solve()
{
  pre_solve();

  if (false_asserted)
    // Then the formula can never be satisfied.
    return smt_convt::P_UNSATISFIABLE;

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
minisat_convt::l_get(const literalt &l)
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
  if (l.is_true())
    return;

  if (l.is_false()) {
    false_asserted = true;
    return;
  }

  Minisat::vec<Lit> c;
  c.push(Minisat::mkLit(l.var_no(), l.sign()));
  solver.addClause_(c);
  return;
}

