#pragma once

/**
 *  Release-checked invariant for goto-symex. Every shipping configuration
 *  compiles with `-DNDEBUG`, so a violated invariant here means the verifier's
 *  own state is inconsistent and any verdict after it rests on nothing.
 *
 *  Reserved for checks no costlier than a comparison — see
 *  docs/roadmap/goto-symex-verification-plan.md, finding R1.
 */
[[noreturn]] void symex_invariant_violated(
  const char *file,
  unsigned line,
  const char *function,
  const char *condition,
  const char *reason);

#define SYMEX_INVARIANT(condition, reason)                                     \
  ((condition) ? static_cast<void>(0)                                          \
               : symex_invariant_violated(                                     \
                   __FILE__, __LINE__, __func__, #condition, (reason)))
