#ifndef UTIL_EXCEPTION_SPECIFICATION_H
#define UTIL_EXCEPTION_SPECIFICATION_H

#include <util/irep.h>
#include <vector>

class typet;

/**
 *  Canonical, function-boundary C++ exception specification.
 *
 *  A C++ exception specification is a property of a function's boundary, not
 *  of any region of its body. It is violated only when exception handler
 *  search exits the function body with an exception the specification does not
 *  permit. This replaces the old executable THROW_DECL/THROW_DECL_END GOTO
 *  instructions, whose enforcement incorrectly depended on the nearest active
 *  try/catch region rather than the function frame.
 *
 *  The canonical representation lives on the function symbol's (legacy) typet
 *  as named-sub attributes so it survives GOTO-binary serialization and is
 *  available for declarations without bodies. goto_functiont caches a decoded
 *  copy for convenient executable access during symbolic execution.
 */
class exception_specificationt
{
public:
  enum class kindt
  {
    /// No specification (or an explicitly throwing one): may throw anything.
    potentially_throwing,
    /// C++11 `noexcept` (or `noexcept(true)`): violation calls std::terminate.
    non_throwing,
    /// Legacy dynamic spec `throw(T...)` (incl. empty `throw()`): violation
    /// calls std::unexpected. `allowed_types` lists the permitted exception
    /// type ids (matching the ids produced by clang_cpp_adjust's
    /// convert_exception_id); empty means nothing may escape.
    dynamic
  };

  exception_specificationt() : kind(kindt::potentially_throwing)
  {
  }

  kindt kind;
  std::vector<irep_idt> allowed_types;

  /// A restrictive spec can be violated at the function boundary. Plain
  /// potentially-throwing functions never violate their spec.
  bool is_restrictive() const
  {
    return kind != kindt::potentially_throwing;
  }

  /// Decode the specification stored on a function typet. Returns a default
  /// (potentially_throwing) spec when no metadata is present.
  static exception_specificationt from_type(const typet &type);

  /// Names of the named-sub attributes used on the typet.
  static const char *kind_attribute()
  {
    return "exception_spec_kind";
  }
  static const char *types_attribute()
  {
    return "exception_spec_types";
  }
};

#endif /* UTIL_EXCEPTION_SPECIFICATION_H */
