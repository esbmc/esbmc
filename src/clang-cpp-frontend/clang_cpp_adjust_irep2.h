#ifndef ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_IREP2_H
#define ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_IREP2_H

#include <clang-c-frontend/clang_c_adjust_irep2.h>

/// IREP2-native adjust pass for the C++ frontend (Phase 7). Mirrors the legacy
/// hierarchy, where clang_cpp_adjust derives from clang_c_adjust and overrides
/// the arms whose C++ behaviour differs.
///
/// It substitutes its own arm table rather than adding rows to the C one: the
/// table is typed on the pass, so this one can order the arms it inherits
/// alongside its own (docs/roadmap/scope-clang-cpp-irep2.md §3.1).
///
/// The table currently lists only inherited arms. That is deliberate and
/// measurable: running it as the sole adjuster over regression/esbmc-cpp
/// enumerates what C++ needs that C does not, which is a measured list where
/// §3's name-mapping was a guess.
class clang_cpp_adjust_irep2 : public clang_c_adjust_irep2
{
public:
  using clang_c_adjust_irep2::clang_c_adjust_irep2;

  /// The arms in application order, as `arm_order()` is for the C pass.
  static std::vector<arm_info> arm_order();

protected:
  void adjust_sole_arms(expr2tc &expr) override;

  /// IREP2 form of clang_cpp_adjust::adjust_cpp_member. `OBJECT.setX()` reaches
  /// the pass with a code-typed member as its callee; goto_convert accepts only
  /// a symbol or a dereference there, so the member is replaced by the symbol
  /// naming the method. The object is already the call's first argument, put
  /// there by the converter, so only the callee changes.
  void adjust_cpp_member(expr2tc &expr);

  /// IREP2 form of clang_cpp_adjust::align_se_function_call_return_type: a call
  /// evaluates to its callee's return type. Constructors are excluded — their
  /// "return type" names the class, not the call's value.
  void align_call_return_type(expr2tc &expr, const symbolt &callee) override;

private:
  using arm = adjust_arm<clang_cpp_adjust_irep2>;

  /// Defined in clang_cpp_adjust_irep2.cpp. Constant-initialised, as the C
  /// table is.
  static const arm arms[];
};

#endif
