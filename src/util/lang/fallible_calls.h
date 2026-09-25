#pragma once

#include <irep2/irep2.h>
#include <string_view>
#include <vector>

/// Fixed whitelist of C-stdlib / POSIX functions whose return value carries
/// success/failure information, used by the CWE-252 unchecked-return-value
/// checker (`--unchecked-return-value-check`).
///
/// Each entry pairs a function's *unmangled* C identifier (matched against
/// the called symbol's pretty name) with the kind of predicate that
/// distinguishes the success domain from the failure domain. The whitelist
/// is intentionally fixed for v1; user-supplied extensions and `errno`-style
/// domains are tracked under the open design questions in issue #4494.

enum class success_kind : uint8_t
{
  /// Success iff the returned pointer is non-null (e.g. `malloc`, `fopen`,
  /// `getenv`).
  non_null,
  /// Success iff the returned integer is non-negative (e.g. `read`,
  /// `write`, `open`).
  non_negative,
  /// Success iff the returned integer is zero (e.g. `pthread_mutex_lock`).
  zero,
};

struct fallible_call_t
{
  std::string_view name;
  success_kind kind;
};

/// Returns the immutable whitelist of fallible calls recognised by v1.
const std::vector<fallible_call_t> &fallible_calls();

/// Looks up a fallible-call entry by unmangled function name. Returns
/// nullptr if `name` is not on the whitelist.
const fallible_call_t *find_fallible(std::string_view name);

/// Builds the boolean expression that holds iff `ret_value` (typed to match
/// the call's return type) lies in the call's success domain.
expr2tc success_predicate(success_kind kind, const expr2tc &ret_value);
