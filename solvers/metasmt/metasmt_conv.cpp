#include "metasmt_conv.h"

#include <solvers/prop/prop_conv.h>

#include <metaSMT/DirectSolver_Context.hpp>
#include <metaSMT/frontend/Logic.hpp>
#include <metaSMT/API/Assertion.hpp>
#include <metaSMT/Instantiate.hpp>
#include <metaSMT/backend/Z3_Backend.hpp>

// To avoid having to build metaSMT into multiple files,
prop_convt *
create_new_metasmt_solver(bool int_encoding, bool is_cpp, const namespacet &ns)
{
  return new metasmt_convt(int_encoding, is_cpp, ns);
}

typedef metaSMT::DirectSolver_Context< metaSMT::solver::Z3_Backend > solvertype;
solvertype ctx;

metasmt_convt::metasmt_convt(bool int_encoding, bool is_cpp,
                             const namespacet &ns)
  : smt_convt(false, int_encoding, ns, is_cpp, false)
{

  metaSMT::assertion(ctx, metaSMT::logic::False);
  std::cerr << "lololol" << metaSMT::solve(ctx) << std::endl;
  abort();
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
                           const smt_ast **args, unsigned int numargs)
{
  abort();
}

smt_sort *
metasmt_convt::mk_sort(const smt_sort_kind k, ...)
{
  abort();
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
  abort();
}

smt_ast *
metasmt_convt::mk_smt_bool(bool val)
{
  abort();
}

smt_ast *
metasmt_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  abort();
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
