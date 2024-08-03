#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/std_expr.h>
#include "util/expr_util.h"
#include "util/cpp_data_object.h"

void clang_cpp_adjust::gen_vbotptr_initializations(symbolt &symbol)
{
  /*
   * This function traverses the vbotptr components of the correponding class type
   * in which the constructor is declared. Then complete the initializations
   * for vbotptrs. A vbotptr is initialized like this:
   *
   * For a standalone class BLAH:
   *  this->BLAH@vbase_offset_ptr = vbase_offset_table::BLAH@BLAH;
   *
   * For a class BLEH derived from BLAH:
   *  this->BLAH@vbotptr = &vtable::BLAH@BLEH;
   *  this->BLEH@vbotptr = &vtable::BLEH@BLEH;
   *
   *  where vtable::BLEH@BLEH contains overriding function,
   *  and vtable::BLAH@BLEH contains the function pointers to
   *  the thunk function that does last casting and redirects the call to the overriding function
   *
   *  N.B. For a derived class, this function assumes that all vbotptrs are properly
   *  copied to the `components` list in its type, which should have been
   *  done in the converter.
   */
  if (!symbol.value.need_vbotptr_init() || symbol.value.is_nil())
    return;

  /*
   * The "to_code_*" conversion functions have done some sanity checks for us:
   *  - a ctor symbol shall contain a function body modelled by code_blockt
   *  - symbol should be of code type
   */
  code_typet &ctor_type = to_code_type(symbol.type);
  code_blockt &ctor_body = to_code_block(to_code(symbol.value));

  /*
   *  vbotptr initializations shall be done in ctor
   *  TODO: For the time being, we just add vbotptr init code in ctors.
   *        We *might* need to add vbotptr init code in dtors in the future. But we need some TCs first.
   */
  if (ctor_type.return_type().id() != "constructor")
    return;

  // get the class' type where this ctor is declared
  const symbolt *ctor_class_symb =
    namespacet(context).lookup(ctor_type.get("#member_name"));
  assert(ctor_class_symb);
  // get the `components` vector from this class' type
  const struct_typet::componentst &components =
    to_struct_type(ctor_class_symb->type).components();

  exprt base;
  get_this_ptr_symbol(ctor_type, base);
  assert(base.type().is_pointer());
  // iterate over the `components` and initialize each virtual pointers
  std::vector<codet> assignments;
  code_blockt assignments_block;
  // iterate over the `components` and initialize each virtual pointers
  handle_components_of_data_object(
    ctor_type, ctor_body, components, base, assignments_block);
  symbol.value.need_vbotptr_init(false);

  symbolt *s =
    context.find_symbol(ctor_type.arguments().back().get("#identifier"));
  assert(ctor_type.arguments().back().get_base_name() == "__is_complete");
  const symbolt &is_complete_symbol = *s;
  assert(s);
  exprt is_complete_symbol_expr = symbol_expr(is_complete_symbol);

  code_ifthenelset assign_vbotptrs_if_complete = code_ifthenelset();
  assign_vbotptrs_if_complete.cond() = is_complete_symbol_expr;
  assign_vbotptrs_if_complete.then_case() = assignments_block;
  ctor_body.operands().insert(
    ctor_body.operands().begin(), assign_vbotptrs_if_complete);

  symbol.value.need_vbotptr_init(false);
}

void clang_cpp_adjust::handle_components_of_data_object(
  const code_typet &ctor_type,
  code_blockt &ctor_body,
  const struct_union_typet::componentst &components,
  exprt &base,
  code_blockt &assignments_block)
{
  for (const auto &data_object_comp_or_vbotptr_comp : components)
  {
    if (data_object_comp_or_vbotptr_comp.get_bool("is_vbot_ptr"))
    {
      side_effect_exprt new_code("assign");
      gen_vbotptr_init_code(
        data_object_comp_or_vbotptr_comp, base, new_code, ctor_type);
      codet code_expr("expression");
      code_expr.move_to_operands(new_code);
      assignments_block.copy_to_operands(code_expr);
      return;
    }
    else if (has_suffix(
               data_object_comp_or_vbotptr_comp.name(),
               cpp_data_object::data_object_suffix))
    {
      assert(has_suffix(
        data_object_comp_or_vbotptr_comp.name(),
        cpp_data_object::data_object_suffix));
      assert(data_object_comp_or_vbotptr_comp.type().is_symbol());
      const typet &data_object_symbol_type =
        data_object_comp_or_vbotptr_comp.type();
      symbolt *data_object_symbol =
        context.find_symbol(data_object_symbol_type.identifier());
      assert(data_object_symbol);

      exprt new_base;
      get_ref_to_data_object(base, data_object_comp_or_vbotptr_comp, new_base);

      struct_typet data_object_type = to_struct_type(data_object_symbol->type);
      handle_components_of_data_object(
        ctor_type,
        ctor_body,
        data_object_type.components(),
        new_base,
        assignments_block);
    }
  }
}

void clang_cpp_adjust::gen_vbotptr_init_code(
  const struct_union_typet::componentt &vbot_comp,
  const exprt &base,
  side_effect_exprt &new_code,
  const code_typet &ctor_type)
{
  /*
   * Generate the statement to assign each vbotptr the corresponding
   * vtable address, e.g.:
   *  this->vbotptr = &<vtable_struct_var_name>
   */

  // 1. set the type
  new_code.type() = vbot_comp.type();

  // 2. LHS: generate the member pointer dereference expression
  exprt lhs_expr = gen_vbotptr_init_lhs(vbot_comp, base, ctor_type);

  // 3. RHS: generate the address of the target virtual pointer struct
  exprt rhs_expr = gen_vbotptr_init_rhs(vbot_comp, ctor_type);

  // now push them to the assignment statement code
  new_code.operands().push_back(lhs_expr);
  new_code.operands().push_back(rhs_expr);
}

exprt clang_cpp_adjust::gen_vbotptr_init_lhs(
  const struct_union_typet::componentt &vbot_comp,
  const exprt &base,
  const code_typet &ctor_type)
{
  /*
   * Generate the LHS expression for virtual pointer initialization,
   * as in:
   *  this->vbotptr = &<vtable_struct_variable>
   */

  exprt lhs_code;

  // now we can get the member expr for "base->vptr"
  assert(!base.type().is_pointer());
  lhs_code = member_exprt(base, vbot_comp.name(), vbot_comp.type());
  lhs_code.set("#lvalue", true);

  return lhs_code;
}

exprt clang_cpp_adjust::gen_vbotptr_init_rhs(
  const struct_union_typet::componentt &comp,
  const code_typet &ctor_type)
{
  /*
   * Generate the RHS expression for virtual pointer initialization,
   * as in:
   *  this->vbotptr = &<vtable_struct_variable>
   */

  exprt rhs_code;

  // get the corresponding vtable variable symbol
  std::string vtable_var_id = comp.type().subtype().identifier().as_string() +
                              "@" + ctor_type.get("#member_name").as_string();
  const symbolt *vtable_var_symb = namespacet(context).lookup(vtable_var_id);
  assert(vtable_var_symb);

  // get the operand for address_of expr as in `&<vtable_struct_variable>`
  exprt vtable_var = symbol_exprt(vtable_var_symb->id, vtable_var_symb->type);
  vtable_var.name(vtable_var_symb->name);

  // now we can get the address_of expr for "&<vtable_struct_variable>"
  rhs_code = address_of_exprt(vtable_var);

  return rhs_code;
}
