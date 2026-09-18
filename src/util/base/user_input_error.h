#ifndef ESBMC_UTIL_BASE_USER_INPUT_ERROR_H
#define ESBMC_UTIL_BASE_USER_INPUT_ERROR_H

#include <exception>

/// Thrown where ESBMC rejects the user's input after reporting why. Caught in
/// main(), which exits with `exit_code`. Throwers log their own diagnostic
/// first, so the message still reaches the user on the paths that swallow the
/// exception (the --k-induction-parallel children report "no answer" and exit).
/// Replaces abort(), whose core dump read as a crash in the verifier rather
/// than a rejected input (esbmc/esbmc#7901).
class user_input_errort : public std::exception
{
public:
  /// Matches the status the other rejected inputs already exit with, e.g. a
  /// GOTO program that could not be built (esbmc_parseoptionst::doit).
  static constexpr int exit_code = 6;

  const char *what() const noexcept override
  {
    return "input rejected";
  }
};

#endif
