// convert_ast must not descend into nil operand slots.
//
// Several expression kinds carry optional operands that are nil -- a
// sideeffect2t built by gen_nondet leaves both `operand` and `size` unset.
// Both conversion paths used to hand those slots on regardless: the work
// stack in convert_ast pushed them and then hashed a null container into
// smt_cache, and convert_ast_node's default case passed them to convert_ast
// via foreach_operand. Either way the result was a SIGSEGV.
//
// The contract is that an unconvertible kind is reported, not crashed on, so
// the test forks and requires SIGABRT (the diagnostic path) rather than
// SIGSEGV. Unreachable from C input, where such a sideeffect is removed
// before symex ends, so only a direct call states it.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <memory>

#include <irep2/irep2_utils.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>
#include <util/config/config.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

#if !defined(_WIN32)
#  include <csignal>
#  include <sys/wait.h>
#  include <unistd.h>

SCENARIO("converting a nil operand slot is reported, not crashed on",
         "[solvers][smt]")
{
  config.ansi_c.set_data_model(configt::LP64);

  GIVEN("a sideeffect whose operand and size slots are both nil")
  {
    // gen_nondet leaves every optional slot unset.
    const expr2tc nondet = gen_nondet(get_int32_type());
    REQUIRE(is_sideeffect2t(nondet));
    REQUIRE(is_nil_expr(to_sideeffect2t(nondet).operand));
    REQUIRE(is_nil_expr(to_sideeffect2t(nondet).size));

    THEN("convert_ast aborts with a diagnostic instead of segfaulting")
    {
      // The child marks the pipe once the solver exists and it is about to
      // call convert_ast, so an abort in create_solver cannot satisfy this.
      int fds[2];
      REQUIRE(pipe(fds) == 0);

      pid_t pid = fork();
      REQUIRE(pid >= 0);
      if (pid == 0)
      {
        close(fds[0]);
        // Silence the diagnostic and Catch2's signal report. Consume the
        // freopen results to satisfy -Werror=unused-result.
        if (
          !freopen("/dev/null", "w", stderr) ||
          !freopen("/dev/null", "w", stdout))
          _exit(2);
        contextt ctx;
        namespacet ns(ctx);
        optionst options;
        std::unique_ptr<smt_convt> solver{create_solver("", ns, options)};
        const char mark = 'x';
        if (write(fds[1], &mark, 1) != 1)
          _exit(3);
        (void)solver->convert_ast(nondet);
        _exit(0); // unreachable: the kind is unsupported
      }

      close(fds[1]);
      char mark = 0;
      const ssize_t got = read(fds[0], &mark, 1);
      close(fds[0]);

      int status = 0;
      REQUIRE(waitpid(pid, &status, 0) == pid);
      // Reached convert_ast: the abort below is its doing, not the solver's.
      REQUIRE(got == 1);
      REQUIRE(mark == 'x');
      REQUIRE(WIFSIGNALED(status));
      // The point of the fix: SIGABRT (reported) rather than SIGSEGV (crash).
      REQUIRE(WTERMSIG(status) != SIGSEGV);
      REQUIRE(WTERMSIG(status) == SIGABRT);
    }
  }
}
#endif
