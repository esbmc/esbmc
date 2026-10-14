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
  void gen_symbol_code(symbolt &symbol) override;

  /// IREP2 form of clang_cpp_adjust::gen_implicit_union_copy_move_constructor.
  /// A union's implicitly-defined copy/move constructor copies the object
  /// representation ([class.copy.ctor]/14), which here is one assignment of the
  /// whole union. Generated before the value walk so the assignment goes
  /// through the arms like any other statement
  /// (docs/roadmap/scope-clang-cpp-irep2.md §8.2).
  void gen_implicit_union_copy_move_body(symbolt &symbol);
  void adjust_symbol_type(symbolt &symbol) override;

  /// IREP2 form of clang_cpp_adjust::adjust_switch's declaration case. C++ lets
  /// a switch condition be a declaration -- `switch (int x = 0)` -- and the
  /// declaration has to be hoisted ahead of the switch, which then switches on
  /// the declared symbol. Left in place the declaration *is* the switched value
  /// and reaches the solver as a statement
  /// (docs/roadmap/scope-clang-cpp-irep2.md §8.1).
  void hoist_switch_declaration(expr2tc &expr);

  void adjust_sole_arms(expr2tc &expr) override;

  /// IREP2 form of clang_cpp_adjust::adjust_cpp_member. `OBJECT.setX()` reaches
  /// the pass with a code-typed member as its callee; goto_convert accepts only
  /// a symbol or a dereference there, so the member is replaced by the symbol
  /// naming the method. The object is already the call's first argument, put
  /// there by the converter, so only the callee changes.
  void adjust_cpp_member(expr2tc &expr);

  /// IREP2 form of clang_cpp_adjust::adjust_catch's id assignment and of the
  /// throw arm's. Both nodes hold their catchable-type ids in an
  /// `exception_list` field, which the converter leaves empty;
  /// remove_exceptions dereferences it, so an unpopulated list is a crash
  /// rather than a lost property (docs/roadmap/scope-clang-cpp-irep2.md §3.3).
  void adjust_cpp_catch(expr2tc &expr);

  /// IREP2 form of clang_cpp_adjust::adjust_cpp_delete: attach the destructor
  /// call `delete p` makes, so goto_convert emits `~T(&(*p))`. Without it the
  /// object's destructors never run (scope-clang-cpp-irep2.md §3.14). The call
  /// travels in sideeffect2t::arguments[0], which the seam already carries.
  void adjust_cpp_delete(expr2tc &expr);

  /// IREP2 form of clang_cpp_adjust::adjust_reference: read each operand that
  /// is a reference through, so it is used as the value it names rather than as
  /// the pointer IREP2 spells it with (scope-clang-cpp-irep2.md §3.16).
  void adjust_reference(expr2tc &expr) override;
  void adjust_cpp_throw(expr2tc &expr);

  /// IREP2 form of clang_cpp_adjust::align_se_function_call_return_type: a call
  /// evaluates to its callee's return type. Constructors are excluded — their
  /// "return type" names the class, not the call's value.
  void align_call_return_type(expr2tc &expr, const symbolt &callee) override;

  /// IREP2 form of clang_cpp_adjust::adjust_side_effect_assign's constructor
  /// fold: `obj = C(args)` becomes the bare call `C(&obj, args)`. Left as an
  /// assignment, remove_sideeffects materialises a temporary for the call's
  /// value and that temporary acquires destructors of its own
  /// (docs/roadmap/scope-clang-cpp-irep2.md §3.14). Runs before the operand
  /// walk: the call-site arms must see the object argument, since they convert
  /// each argument against the matching parameter and a constructor's first
  /// parameter is `this`. Legacy asserts the callee is a constructor and that
  /// the object roots at a symbol; both are declines here, since a pass that
  /// substitutes for the legacy one should leave a node it does not understand
  /// alone rather than abort on it.
  void adjust_before_operands(expr2tc &expr) override;
  void fold_constructor_assignment(expr2tc &expr);

  /// IREP2 form of clang_cpp_adjust::adjust_decl_block's fan-out: a local array
  /// of class type arrives with one constructor call for the whole array, and
  /// every element has to be constructed. Operates on the enclosing block
  /// because one declaration becomes several statements; rewriting the
  /// declaration into a block of its own would end the object's scope at that
  /// block's brace (esbmc/esbmc#4715).
  void fan_out_array_construction(expr2tc &expr);

  /// Nil unless `stmt` declares a local array whose initialiser is a single
  /// whole-array constructor call; otherwise that call.
  expr2tc array_decl_constructor(const expr2tc &stmt);
  /// False when a dimension's size is not a constant, leaving `out` unusable:
  /// the caller emits nothing in that case.
  bool construct_elements(
    const expr2tc &array,
    const expr2tc &ctor,
    std::vector<expr2tc> &out);

  /// Whether `call` is a constructor call. The converter's `#constructor`
  /// marker does not cross the seam, so the `constructor` return-type spelling
  /// is read from the symbol table instead. The two are not the same set --
  /// legacy sets the marker only where it built the call -- but every shape
  /// measured so far wants the same answer (§3.16).
  bool is_constructor_call(const expr2tc &call);
  expr2tc find_constructor_call(const expr2tc &e);

private:
  using arm = adjust_arm<clang_cpp_adjust_irep2>;

  /// Defined in clang_cpp_adjust_irep2.cpp. Constant-initialised, as the C
  /// table is.
  static const arm arms[];
};

#endif
