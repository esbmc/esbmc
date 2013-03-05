#include <unistd.h>

#include "smtlib_conv.h"

smtlib_convt::smtlib_convt(bool int_encoding, const namespacet &_ns,
                           bool is_cpp, const optionst &_opts)
  : smt_convt(false, int_encoding, _ns, is_cpp, false), options(_opts)
{
  // Setup: open a pipe to the smtlib solver. Because C++ is terrible,
  // there's no standard way of opening a stream from an fd, we can try
  // a nonportable way in the future if fwrite becomes unenjoyable.

  int inpipe[2], outpipe[2];
  if (pipe(inpipe) != 0) {
    std::cerr << "Couldn't open a pipe for smtlib solver" << std::endl;
    abort();
  }

  if (pipe(outpipe) != 0) {
    std::cerr << "Couldn't open a pipe for smtlib solver" << std::endl;
    abort();
  }

  pid_t pid = fork();
  if (pid == 0) {
    abort();
  } else {
    abort();
  }
}

smtlib_convt::~smtlib_convt()
{
}

prop_convt::resultt
smtlib_convt::dec_solve()
{
  abort();
}

expr2tc
smtlib_convt::get(const expr2tc &expr __attribute__((unused)))
{
  abort();
}

tvt
smtlib_convt::l_get(literalt a __attribute__((unused)))
{
  abort();
}

const std::string
smtlib_convt::solver_text()
{
  abort();
}

void
smtlib_convt::assert_lit(const literalt &l __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_func_app(const smt_sort *s __attribute__((unused)), smt_func_kind k __attribute__((unused)),
                          const smt_ast **args __attribute__((unused)), unsigned int numargs __attribute__((unused)))
{
  abort();
}

smt_sort *
smtlib_convt::mk_sort(const smt_sort_kind k __attribute__((unused)), ...)
{
  abort();
}

literalt
smtlib_convt::mk_lit(const smt_ast *s __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_smt_int(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_smt_real(const mp_integer &thereal __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_smt_bvint(const mp_integer &theint __attribute__((unused)), bool sign __attribute__((unused)), unsigned int w __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_smt_bool(bool val __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_smt_symbol(const std::string &name __attribute__((unused)), const smt_sort *s __attribute__((unused)))
{
  abort();
}

smt_sort *
smtlib_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_sort *
smtlib_convt::mk_union_sort(const type2tc &type __attribute__((unused)))
{
  abort();
}

smt_ast *
smtlib_convt::mk_extract(const smt_ast *a __attribute__((unused)), unsigned int high __attribute__((unused)), unsigned int low __attribute__((unused)),
                         const smt_sort *s __attribute__((unused)))
{
  abort();
}
