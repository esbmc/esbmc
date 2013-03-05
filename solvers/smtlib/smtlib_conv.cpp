#include <unistd.h>

#include "smtlib_conv.h"
#include "y.tab.hpp"

// Dec of external lexer input stream
extern "C" FILE *smtlibin;
int smtlibparse(int startval);
extern int smtlib_send_start_code;

smtlib_convt::smtlib_convt(bool int_encoding, const namespacet &_ns,
                           bool is_cpp, const optionst &_opts)
  : smt_convt(false, int_encoding, _ns, is_cpp, false), options(_opts)
{
  // Setup: open a pipe to the smtlib solver. Because C++ is terrible,
  // there's no standard way of opening a stream from an fd, we can try
  // a nonportable way in the future if fwrite becomes unenjoyable.

  int inpipe[2], outpipe[2];
  std::string cmd;

  cmd = options.get_option("smtlib-solver-prog");
  if (cmd == "") {
    std::cerr << "Must specify an smtlib solver program in smtlib mode"
              << std::endl;
    abort();
  }

  if (pipe(inpipe) != 0) {
    std::cerr << "Couldn't open a pipe for smtlib solver" << std::endl;
    abort();
  }

  if (pipe(outpipe) != 0) {
    std::cerr << "Couldn't open a pipe for smtlib solver" << std::endl;
    abort();
  }

  solver_proc_pid = fork();
  if (solver_proc_pid == 0) {
    close(outpipe[1]);
    close(inpipe[0]);
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    dup2(outpipe[0], STDIN_FILENO);
    dup2(inpipe[1], STDOUT_FILENO);
    close(outpipe[0]);
    close(inpipe[1]);

    // Voila
    execlp(cmd.c_str(), cmd.c_str(), NULL);
    std::cerr << "Exec of smtlib solver failed" << std::endl;
    abort();
  } else {
    close(outpipe[0]);
    close(inpipe[1]);
    out_stream = fdopen(outpipe[1], "w");
    in_stream = fdopen(inpipe[0], "r");
  }

  // Execution continues as the parent ESBMC process. Child dying will
  // trigger SIGPIPE or an EOF eventually, which we'll be able to detect
  // and crash upon.

  // Point lexer input at output stream
  smtlibin = in_stream;

  // Fetch solver name and version.
  fprintf(out_stream, "(get-info :name)\n");
  fflush(out_stream);
  smtlib_send_start_code = 1;
  unsigned int ret = smtlibparse(TOK_START_INFO);
  std::cerr << "Ohai, return code was " << ret << std::endl;
  abort();

  fprintf(out_stream, "(get-info :version)\n");
  fflush(out_stream);
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

int
smtliberror(int startsym __attribute__((unused)), const std::string &error)
{
  std::cerr << "SMTLIB response parsing error: \"" << error << "\""
            << std::endl;
  abort();
}
