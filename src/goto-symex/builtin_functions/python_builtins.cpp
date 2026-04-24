#include <goto-symex/goto_symex.h>
#include <string>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/std_types.h>
#include <algorithm>

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
    cur_state->value_set.get_value_set(value, value_set);

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
    // Remove all typecast to get the original type
    while (is_typecast2t(value))
      value = to_typecast2t(value).from;

    if (is_address_of2t(value))
      value = to_address_of2t(value).ptr_obj;

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
      if (base_type_eq(expect_type->type, value->type, ns))
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
      // Special case, str is modeled as a array, we need to get its subtype
      t = to_index2t(value).source_value->type;
    else
      t = value->type;

    if (base_type_eq(t, expect_type->type, ns))
      expr = gen_true_expr();
    else
      expr = gen_false_expr();

    if (!is_nil_expr(expect_type) && is_array_type(expect_type->type))
    {
      // In the memory model, an array of size 1 is simplified to a single element
      // Therefore, here we specifically check whether the subtypes of the arrays are the same
      // s:str = "" ----> 0 with char type
      // This should be safe because int, bool and char have different widths,
      // so there will be no confusion
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

    type2tc obj_type = value->type;
    if (is_pointer_type(obj_type))
      obj_type = to_pointer_type(obj_type).subtype;

    if (is_struct_type(obj_type))
    {
      const struct_type2t &st = to_struct_type(obj_type);
      const auto &members = st.get_structure_member_names();
      const bool has_member =
        std::any_of(members.begin(), members.end(), [&](const irep_idt &memb) {
          return memb.as_string() == attr_name;
        });
      expr = has_member ? gen_true_expr() : gen_false_expr();
    }
    else
      expr = gen_false_expr();

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
        // For void* (any_type) with no Optional in the value set,
        // fall back to null-pointer check: x is None ↔ x == NULL.
        if (is_empty_type(to_pointer_type(side->type).subtype))
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
