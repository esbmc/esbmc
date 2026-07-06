#include <functional>
#include <clang-cpp-frontend/clang_cpp_main.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/namespace.h>
#include <util/std_expr.h>
#include <util/symbolic_types.h>

clang_cpp_maint::clang_cpp_maint(contextt &_context) : clang_c_maint(_context)
{
}

// Recursively locate the constructor-call side-effect inside an initializer
// tree.  For a scalar object the call is the immediate operand of the
// temporary_object's `initializer()` code; for an array the frontend wraps it
// as `&ctor(...)[0]`, so the call sits a few levels down (address_of -> index
// -> sideeffect).  Returns nullptr if none is present.
static const exprt *find_constructor_call(const exprt &e)
{
  if (
    e.id() == "sideeffect" && e.statement() == "function_call" &&
    e.get_bool("constructor"))
    return &e;

  forall_operands (it, e)
    if (const exprt *found = find_constructor_call(*it))
      return found;

  return nullptr;
}

void clang_cpp_maint::adjust_init(code_assignt &assignment, codet &adjusted)
{
  // adjust the init statement for global variables
  assert(assignment.operands().size() == 2);

  exprt &rhs = assignment.rhs();

  // For class-type globals initialised by a constructor call, the C++ frontend
  // wraps the constructor call in a `sideeffect/temporary_object` whose
  // `initializer()` sub-irep carries a `codet("expression")` containing the
  // actual call (see clang_cpp_convertert::make_temporary).  Unwrap it here so
  // the rest of this routine can handle it like a direct constructor call.
  if (rhs.id() == "sideeffect" && rhs.statement() == "temporary_object")
  {
    const irept &init_irep = rhs.initializer();
    if (init_irep.is_not_nil())
    {
      const exprt &init_expr = static_cast<const exprt &>(init_irep);
      if (const exprt *ctor = find_constructor_call(init_expr))
      {
        exprt unwrapped = *ctor;
        rhs.swap(unwrapped);
      }
    }
  }

  if (
    rhs.id() == "sideeffect" && rhs.statement() == "function_call" &&
    rhs.get_bool("constructor"))
  {
    // Per [basic.start.static]/2, static-storage-duration objects are
    // zero-initialized before any other initialization takes place.
    // Constructors for non-aggregate class types only assign the members they
    // explicitly mention, so emit an explicit zero-assignment here so any
    // member the constructor leaves untouched still reads as zero.  Fixes the
    // long-standing bug where, e.g., `struct C { int x; C(){} } c;` had `c.x`
    // show up as nondet.
    //
    // Use code_assignt rather than code_declt: globals already have storage,
    // so a decl is dropped by goto-conversion and the zero-init would be lost.
    namespacet ns(context);
    exprt zero = gen_zero(get_complete_type(assignment.lhs().type(), ns), true);
    if (zero.is_not_nil())
    {
      // Guard against gen_zero returning nil for unresolved/dependent types
      // (e.g. a global of an incomplete type).  Skipping the zero-init in
      // that case preserves prior behaviour rather than enqueuing a malformed
      // ASSIGN.
      code_assignt zero_init(assignment.lhs(), zero);
      zero_init.location() = assignment.location();
      adjusted.copy_to_operands(zero_init);
    }

    // Get rhs - this represents the constructor call
    side_effect_expr_function_callt &init =
      to_side_effect_expr_function_call(rhs);

    // Get lhs - this represents the `this` pointer.  The original lhs needs to
    // be the first arg, then followed by others: BLAH(&bleh, arg1, arg2, ...).
    if (ns.follow(assignment.lhs().type()).is_array())
    {
      // Array of class type: the single CXXConstructExpr stands for
      // constructing every element, so emit one constructor call per element
      // with `this` pointing at that element (`&arr[i]`).  Recurse into nested
      // arrays so multidimensional arrays construct every leaf element.
      std::function<void(const exprt &)> construct_elements =
        [&](const exprt &arr) {
          const array_typet &arr_type = to_array_type(ns.follow(arr.type()));
          BigInt count;
          if (to_integer(arr_type.size(), count))
          {
            log_error("cannot determine array size for static ctor init");
            abort();
          }

          const typet &elem_type = arr_type.subtype();
          for (BigInt idx = 0; idx < count; ++idx)
          {
            exprt element =
              index_exprt(arr, from_integer(idx, index_type()), elem_type);
            if (ns.follow(elem_type).is_array())
            {
              construct_elements(element);
              continue;
            }
            side_effect_expr_function_callt elem_init = init;
            elem_init.arguments()[0] = address_of_exprt(element);
            convert_expression_to_code(elem_init);
            adjusted.copy_to_operands(elem_init);
          }
        };
      construct_elements(assignment.lhs());
    }
    else
    {
      init.arguments()[0] = address_of_exprt(assignment.lhs());

      // Now convert the side_effect into an expression
      convert_expression_to_code(init);

      // and copy to adjusted
      adjusted.copy_to_operands(init);
    }
  }
}

void clang_cpp_maint::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}
