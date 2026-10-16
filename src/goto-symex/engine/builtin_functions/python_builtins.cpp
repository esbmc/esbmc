#include <goto-symex/engine/goto_symex.h>
#include <string>
#include <util/arith/arith_tools.h>
#include <util/expr/base_type.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>
#include <util/irep/migrate.h>
#include <util/irep/std_types.h>
#include <algorithm>

/// Whether a NULL pointer to @p subt is how the frontend spells Python None.
static bool null_is_none(const type2tc &subt, const namespacet &ns)
{
  if (is_empty_type(subt) || is_struct_type(subt) || is_array_type(subt))
    return true;
  return is_symbol_type(subt) &&
         ns.follow(migrate_type_back(subt)).id() == "struct";
}

/// Whether the Python class @p tag, or one of its bases, defines a method
/// @p name. The class type does not keep its method list, so this looks for
/// the method's function symbol, which the frontend names
/// py:<module>@C@<class>@F@<method>. The constructor is named after the class.
static bool class_has_method(
  const irep_idt &tag,
  const std::string &name,
  const namespacet &ns)
{
  const symbolt *cls = ns.lookup(tag);
  if (!cls || !cls->get_type().is_struct())
    return false;
  const std::string class_name = cls->name.as_string();
  if (name != class_name)
  {
    const symbolt *method = ns.lookup(
      "py:" + cls->module.as_string() + "@C@" + class_name + "@F@" + name);
    if (method && method->get_type().is_code())
      return true;
  }
  forall_irep (base, cls->get_type().find("bases").get_sub())
    if (class_has_method(base->id(), name, ns))
      return true;
  return false;
}

/// Whether a scalar of type @p type is an instance of the Python type that
/// @p expected represents. bool is a subclass of int.
static bool is_python_scalar_instance(
  const type2tc &type,
  const type2tc &expected,
  const namespacet &ns)
{
  return base_type_eq(type, expected, ns) ||
         (is_bool_type(type) && expected == long_long_int_type2());
}

/// Whether an object of type @p type has the attribute @p name: an instance
/// member or a method of its class.
static bool has_python_attribute(
  type2tc type,
  const std::string &name,
  const namespacet &ns)
{
  if (is_pointer_type(type))
    type = to_pointer_type(type).subtype;
  if (!is_struct_type(type))
    return false;
  const auto &members = struct_union_member_names(type);
  return std::any_of(
           members.begin(),
           members.end(),
           [&](const irep_idt &memb) { return memb.as_string() == name; }) ||
         class_has_method(
           "tag-" + to_struct_type(type).name.as_string(), name, ns);
}

/// Whether @p value is exactly class @p expected. Classes are told apart by
/// name, as two distinct classes may share a layout (both empty, say).
static bool
same_class(const type2tc &expected, const type2tc &value, const namespacet &ns)
{
  if (is_struct_type(expected))
    return to_struct_type(expected).name == to_struct_type(value).name;
  return base_type_eq(expected, value, ns);
}

/// Whether @p type is how a Python int, float or bool is held.
static bool is_python_number_type(const type2tc &type)
{
  return is_floatbv_type(type) || is_bool_type(type) ||
         (is_signedbv_type(type) && type->get_width() == 64);
}

/// The value isinstance inspects: the one traced from @p operand, with its
/// casts stripped to reach the object a pointer refers to. A number converted
/// to another number is answered from @p operand's own type instead:
/// `a = float(x)` holds `(double)x`, whose operand is an int.
static expr2tc isinstance_subject(expr2tc value, const expr2tc &operand)
{
  while (is_typecast2t(value))
    value = to_typecast2t(value).from;
  return is_python_number_type(operand->type) &&
             is_python_number_type(value->type)
           ? operand
           : value;
}

// An untyped (`void *`) operand that neither constant propagation nor the
// value set traced back to its object carries no Python type, so no answer is
// sound: comparing `void *` itself would say "no" to every type. Fail only if
// this path is reachable, and end it so no check after it relies on the guess.
void goto_symext::reject_untyped_isinstance(const expr2tc &value)
{
  if (
    !is_symbol2t(value) || !is_pointer_type(value->type) ||
    !is_empty_type(to_pointer_type(value->type).subtype))
    return;
  claim(gen_false_expr(), "isinstance() on a value whose type was lost");
  assume(gen_false_expr());
}

void goto_symext::simplify_python_builtins(expr2tc &expr)
{
  expr->Foreach_operand([this](expr2tc &e) {
    if (!is_nil_expr(e))
      simplify_python_builtins(e);
  });

  if (is_isinstance2t(expr))
  {
    const isinstance2t &obj = to_isinstance2t(expr);
    expr2tc value = obj.side_1;
    expr2tc expect_type = obj.side_2;

    // isinstance(None, ...) is always False
    if (is_pointer_type(value->type))
    {
      const pointer_type2t &ptr = to_pointer_type(value->type);
      if (is_bool_type(ptr.subtype))
      {
        if (
          !is_struct_type(expect_type->type) &&
          !is_pointer_type(expect_type->type))
        {
          expr = gen_false_expr();
          return;
        }
      }
    }

    value_setst::valuest value_set;
    // The value-set map is keyed by the L1 symbol name (e.g. "...@x?1!0"); the
    // isinstance operand arrives as an L0 symbol, so an unrenamed lookup misses
    // and the points-to set comes back empty/unknown. Rename to L1 first. This
    // matters for the object-model migration (#3067): a class instance is a
    // `Class*` allocated by `new_object`; when that allocation is guarded
    // (e.g. inside `if c:`), constant propagation cannot recover the concrete
    // object from the renamed pointer, so the value-set is the only way to find
    // the pointed-to struct and resolve the type. The L1 rename keeps the
    // symbol shape (cur_state->rename may wrap it in a typecast), so we can
    // index the map directly.
    expr2tc l1_value = value;
    cur_state->top().level1.rename(l1_value);
    cur_state->value_set.get_value_set(l1_value, value_set);

    // Find the last value from the value set
    for (const auto &obj : value_set)
    {
      if (is_object_descriptor2t(obj))
      {
        const object_descriptor2t &o = to_object_descriptor2t(obj);
        value = o.object;
      }
    }

    cur_state->rename(value);
    value = isinstance_subject(value, obj.side_1);

    if (is_address_of2t(value))
      value = to_address_of2t(value).ptr_obj;

    reject_untyped_isinstance(value);

    if (is_struct_type(value))
    {
      // Check if this is a tuple by examining the tag
      if (is_nil_expr(expect_type))
      {
        // find tuple type
        const struct_type2t &struct_type = to_struct_type(value->type);
        if (struct_type.name.as_string().find("tag-tuple") == 0)
          expr = gen_true_expr();
        else
          expr = gen_false_expr();

        return;
      }

      // Check sub class
      if (same_class(expect_type->type, value->type, ns))
        expr = gen_true_expr();
      else if (
        is_struct_type(expect_type->type) &&
        is_subclass_of(expect_type->type, value->type, ns))
        expr = gen_true_expr();
      else
        expr = gen_false_expr();

      return;
    }

    // Basic type comparison
    // int, str, bool
    type2tc t;
    if (is_index2t(value))
    {
      // str is modeled as an array; also handles typed global arrays such as
      // __ESBMC_float_buf[idx] used by float list elements.
      t = to_index2t(value).source_value->type;
      // When the source is an array, also accept a match against the element
      // type (e.g. double[4096] element type == double for isinstance(x,
      // float)).
      if (
        is_array_type(t) &&
        base_type_eq(to_array_type(t).subtype, expect_type->type, ns))
      {
        expr = gen_true_expr();
        return;
      }
    }
    else
      t = value->type;

    if (is_python_scalar_instance(t, expect_type->type, ns))
      expr = gen_true_expr();
    else
      expr = gen_false_expr();

    if (!is_nil_expr(expect_type) && is_array_type(expect_type->type))
    {
      // In the memory model, an array of size 1 is simplified to a single
      // element Therefore, here we specifically check whether the subtypes of
      // the arrays are the same s:str = "" ----> 0 with char type This should
      // be safe because int, bool and char have different widths, so there will
      // be no confusion
      if (to_array_type(expect_type->type).subtype == value->type)
        expr = gen_true_expr();
    }

    return;
  }
  else if (is_hasattr2t(expr))
  {
    const hasattr2t &obj = to_hasattr2t(expr);
    expr2tc value = obj.side_1;
    expr2tc attr = obj.side_2;

    // Only simplify when the attribute name is a constant string.
    if (!is_constant_string2t(attr))
      return;

    const auto &attr_const = to_constant_string2t(attr);
    std::string attr_name = attr_const.value.as_string();

    cur_state->rename(value);
    while (is_typecast2t(value))
      value = to_typecast2t(value).from;
    if (is_address_of2t(value))
      value = to_address_of2t(value).ptr_obj;

    expr = has_python_attribute(value->type, attr_name, ns) ? gen_true_expr()
                                                            : gen_false_expr();
    return;
  }
  else if (is_isnone2t(expr))
  {
    const isnone2t &cmp = to_isnone2t(expr);
    expr2tc lhs = cmp.side_1;
    expr2tc rhs = cmp.side_2;

    cur_state->rename(lhs);
    cur_state->rename(rhs);

    // Remove typecasts to get original types
    while (is_typecast2t(lhs))
      lhs = to_typecast2t(lhs).from;
    while (is_typecast2t(rhs))
      rhs = to_typecast2t(rhs).from;

    auto is_none_type = [](const expr2tc &e) -> bool {
      // Python None is represented as bool* (none_type()).
      // void* (any_type) is for unannotated variables, not Python None.
      if (is_pointer_type(e))
      {
        const pointer_type2t &ptr_type = to_pointer_type(e->type);
        return is_bool_type(ptr_type.subtype);
      }

      return false;
    };

    const bool lhs_is_none = is_none_type(lhs);
    const bool rhs_is_none = is_none_type(rhs);

    // Check if an expr2tc is an Optional struct and return its is_none field
    auto optional_is_none_field =
      [&](const expr2tc &val) -> std::optional<expr2tc> {
      if (!is_struct_type(val))
        return std::nullopt;
      const struct_type2t &st = to_struct_type(val->type);
      if (!st.name.as_string().starts_with("tag-Optional_"))
        return std::nullopt;
      return member2tc(get_bool_type(), val, "is_none");
    };

    // Handle Optional[T] vs None
    auto handle_optional_side =
      [&](const expr2tc &side, bool other_is_none) -> std::optional<expr2tc> {
      if (!other_is_none)
        return std::nullopt;

      // Direct Optional struct case
      if (auto res = optional_is_none_field(side))
        return res;

      // Pointer case (Optional* or void*): resolve via value set to find the
      // actual Optional struct the pointer points to. We return on the first
      // Optional entry found; for single-assignment variables the value set
      // has exactly one entry. The tag "tag-Optional_" is established by
      // type_handler::build_optional_type.
      if (is_pointer_type(side))
      {
        value_setst::valuest value_set;
        cur_state->value_set.get_value_set(side, value_set);
        for (const auto &obj : value_set)
        {
          if (!is_object_descriptor2t(obj))
            continue;
          expr2tc val = to_object_descriptor2t(obj).object;
          cur_state->rename(val);
          while (is_typecast2t(val))
            val = to_typecast2t(val).from;
          if (auto res = optional_is_none_field(val))
            return res;
        }
        // Fall back to a null-pointer check: `x is None` ↔ `x == NULL`.
        // This covers void* (any_type, unannotated) and — since class
        // instances became heap-allocated Class* references — a pointer to a
        // user-class struct, whose field/variable holds NULL exactly when its
        // Python value is None. The pointee may still be an unresolved symbol
        // tag (`tag-<Class>`) rather than an inlined struct, so follow it.
        // Without this, `Class* is None` folded to a constant False below
        // ("None is never equal to non-None"), silently defeating guards like
        // `node is None or node.next is None` and producing spurious NULL
        // dereferences (QuixBugs detect_cycle). An Optional[str] is a pointer
        // to its char array, NULL for None, as well.
        if (null_is_none(to_pointer_type(side->type).subtype, ns))
          return equality2tc(side, gen_zero(side->type));
      }
      return std::nullopt;
    };

    if (!lhs_is_none)
    {
      if (auto res = handle_optional_side(lhs, rhs_is_none))
      {
        expr = *res;
        return;
      }
    }
    if (!rhs_is_none)
    {
      if (auto res = handle_optional_side(rhs, lhs_is_none))
      {
        expr = *res;
        return;
      }
    }

    // Handle None vs None pointer comparisons (identity check)
    if (lhs_is_none && rhs_is_none)
    {
      const expr2tc &ptr_expr = lhs;
      expr2tc null_ptr = gen_zero(ptr_expr->type);
      expr = equality2tc(ptr_expr, null_ptr);
      return;
    }

    // Handle None vs non-None comparison
    if ((lhs_is_none && !rhs_is_none) || (rhs_is_none && !lhs_is_none))
    {
      // None is never equal to non-None values
      expr = gen_false_expr();
      return;
    }

    // Handle non-None comparisons
    expr = gen_true_expr();
  }
}
