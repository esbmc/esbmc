#ifndef CPROVER_UTIL_FOCUS_FUNCTION_H
#define CPROVER_UTIL_FOCUS_FUNCTION_H

#include <string>
#include <vector>

/// \file focus_function.h
/// \brief The ONE place `--focus-function`'s value is interpreted.
///
/// `--focus-function` names a SET of the target contract's public/external
/// functions, not a single one. Three places have to agree on which functions a
/// given value selects:
///
///   1. the Solidity frontend's validator (`solidity_convertert::convert`),
///      which fails the conversion when a named function does not exist;
///   2. the frontend's dispatcher filter (`get_unbound_function`), which decides
///      which entries the harness may call;
///   3. the path-coverage pass (`goto_coveraget::focus_selects_unit`), which
///      decides which units are ENUMERATED AND INSTRUMENTED, and hence what the
///      published denominator is.
///
/// They live in different modules, so the temptation is three small parsers.
/// That would be a detector keyed on a condition its own branch does not state:
/// if (2) and (3) disagreed, a unit the dispatcher can enter would carry no
/// claims -- an invisible zero in the numerator -- or a unit nothing can enter
/// would sit in the denominator, which is the 45x error the narrowing was
/// written to remove in the first place. One parser, in `util` because both
/// `solidity-frontend` and `goto-programs` already depend on it, makes the
/// disagreement unrepresentable rather than merely unlikely.
///
/// SEPARATORS. Comma OR whitespace, and both may be mixed:
///
///     --focus-function deposit,withdraw
///     --focus-function "deposit withdraw"
///     --focus-function "deposit, withdraw"
///
/// Whitespace is accepted because `--contract` already takes its list that way
/// (`--contract "C1 C2"`, split with `istringstream >>`), and a user who has
/// just written one is entitled to expect the other to behave the same. Comma is
/// accepted because a shell user writing a list without quotes reaches for it
/// first. Empty fields (`a,,b`, a trailing comma) are dropped rather than
/// becoming an empty name that matches nothing and is diagnosed as a typo.
///
/// MATCHING IS EXACT, per name, on the SOURCE-LEVEL function name. It is not a
/// prefix or substring test: `--focus-function pub` must not select `pubx`, and
/// a name-only match deliberately keeps EVERY OVERLOAD of that name, because the
/// dispatcher offers all of them and an entry that can be entered must not be
/// left unmeasured.

/// Split a `--focus-function` value into the function names it selects.
/// Order is preserved (so diagnostics list names as the user wrote them) and
/// duplicates are NOT removed: a repeated name selects the same unit twice,
/// which is harmless, and silently collapsing it would hide a typo of the form
/// `--focus-function deposit,deposit` where the second was meant to be another
/// function.
inline std::vector<std::string> focus_function_names(const std::string &raw)
{
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&out, &cur]() {
    if (!cur.empty())
      out.push_back(cur);
    cur.clear();
  };
  for (char c : raw)
  {
    if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r')
      flush();
    else
      cur.push_back(c);
  }
  flush();
  return out;
}

/// Does `raw` (a whole `--focus-function` value) select the function named
/// `fn`? An EMPTY value selects everything -- that is "no narrowing", the state
/// every caller must treat as "the focus is off", not as "the focus selects
/// nothing".
inline bool
focus_function_selects(const std::string &raw, const std::string &fn)
{
  if (raw.empty())
    return true;
  for (const auto &name : focus_function_names(raw))
    if (name == fn)
      return true;
  return false;
}

#endif
