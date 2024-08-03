#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/std_expr.h>
#include "util/cpp_typecast.h"
#include "util/cpp_data_object.h"

void clang_cpp_adjust::gen_vptr_initializations(symbolt &symbol)
{
  /*
   * This function traverses the vptr components of the correponding class type
   * in which the constructor is declared. Then complete the initializations
   * for vptrs. A vptr is initialized like this:
   *
   * For a standalone class BLAH:
   *  this->BLAH@vptr = &vtable::BLAH@BLAH;
   *
   * For a class BLEH derived from BLAH:
   *  this->BLAH@vptr = &vtable::BLAH@BLEH;
   *  this->BLEH@vptr = &vtable::BLEH@BLEH;
   *
   *  where vtable::BLEH@BLEH contains overriding function,
   *  and vtable::BLAH@BLEH contains the function pointers to
   *  the thunk function that does last casting and redirects the call to the overriding function
   *
   *  N.B. For a derived class, this function assumes that all vptrs are properly
   *  copied to the `components` list in its type, which should have been
   *  done in the converter.
   */
  if (!symbol.value.need_vptr_init() || symbol.value.is_nil())
    return;

  /*
   * The "to_code_*" conversion functions have done some sanity checks for us:
   *  - a ctor symbol shall contain a function body modelled by code_blockt
   *  - symbol should be of code type
   */
  code_typet &ctor_type = to_code_type(symbol.type);
  code_blockt &ctor_body = to_code_block(to_code(symbol.value));

  /*
   *  vptr initializations shall be done in ctor
   *  TODO: For the time being, we just add vptr init code in ctors.
   *        We *might* need to add vptr init code in dtors in the future. But we need some TCs first.
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

  // iterate over the `components` and initialize each virtual pointers
  handle_components_of_data_object(ctor_type, ctor_body, components, base);

  symbol.value.need_vptr_init(false);
}

void clang_cpp_adjust::get_this_ptr_symbol(
  const code_typet &ctor_type,
  exprt &this_ptr)
{
  // get the `this` argument symbol
  const symbolt *this_symb = namespacet(context).lookup(
    ctor_type.arguments().at(0).type().subtype().identifier());
  assert(this_symb);
  assert(this_symb->type.is_struct());

  // create the `this` pointer expr
  this_ptr = symbol_exprt(
    ctor_type.arguments().at(0).get("#identifier"),
    pointer_typet(this_symb->type));
}

void clang_cpp_adjust::get_ref_to_data_object(
  exprt &base,
  const struct_union_typet::componentt &data_object_comp,
  exprt &data_object_ref)
{
  bool is_virtual = data_object_comp.get_bool("from_virtual_base");
  if (is_virtual)
  {
    if (!base.type().is_pointer())
    {
      exprt addr_of_base = address_of_exprt(base);
      addr_of_base.type() = pointer_typet(base.type());
      base.swap(addr_of_base);
    }
    assert(data_object_comp.type().is_symbol());
    // get the `virtual_base` symbol
    std::string virtual_base_id =
      data_object_comp.type().identifier().as_string();
    virtual_base_id = virtual_base_id.substr(
      0, virtual_base_id.find(cpp_data_object::data_object_suffix));
    const symbolt *virtual_base_symb = ns.lookup(virtual_base_id);
    assert(virtual_base_symb);
    assert(virtual_base_symb->type.is_struct());
    typet virtual_base_symbol_type = symbol_typet(virtual_base_id);
    assert(virtual_base_symbol_type.identifier() == virtual_base_symb->id);
    // TODO: This currently casts to "tag-<type>@data_object" type.
    // We need to cast to the actual type of the virtual base.
    // For non-virtual bases we can just directly reference the data object,
    // because we know the layout of the non-virtual data objects.
    // For virtual data objects this is __not__ the case. Therefore, we have to cast
    // to the actual type of the virtual base which will perform any necessary adjustments.
    //    typet dest_type = pointer_typet(virtual_base_symb->type);
    typet dest_type = pointer_typet(virtual_base_symbol_type);
    assert(dest_type.is_pointer());
    assert(dest_type.subtype().is_symbol());
    cpp_typecast::derived_to_base_typecast(base, dest_type, true, ns);
  }
  if (base.type().is_pointer())
  {
    exprt base_deref = dereference_exprt(base, base.type());
    base.swap(base_deref);
  }
  assert(!base.type().is_pointer());
  // now get the data object member expr for "this-><data_object_comp_name>"
  exprt data_object_member =
    member_exprt(base, data_object_comp.name(), data_object_comp.type());
  data_object_member.set("#lvalue", true);
  data_object_ref.swap(data_object_member);
  assert(!base.type().is_pointer());
  //  assert(!data_object_ref.type().is_pointer());
}

void clang_cpp_adjust::handle_components_of_data_object(
  const code_typet &ctor_type,
  code_blockt &ctor_body,
  const struct_union_typet::componentst &components,
  exprt &base)
{
  for (const auto &data_object_comp_or_vptr_comp : components)
  {
    if (data_object_comp_or_vptr_comp.get_bool("is_vtptr"))
    {
      side_effect_exprt new_code("assign");
      gen_vptr_init_code(
        data_object_comp_or_vptr_comp, base, new_code, ctor_type);
      codet code_expr("expression");
      code_expr.move_to_operands(new_code);
      ctor_body.operands().push_back(code_expr);
      return;
    }
    else if (has_suffix(
               data_object_comp_or_vptr_comp.name(),
               cpp_data_object::data_object_suffix))
    {
      assert(has_suffix(
        data_object_comp_or_vptr_comp.name(),
        cpp_data_object::data_object_suffix));
      //      assert(data_object_comp_or_vptr_comp.type().is_struct());
      assert(data_object_comp_or_vptr_comp.type().is_symbol());
      const typet &data_object_symbol_type =
        data_object_comp_or_vptr_comp.type();
      symbolt *data_object_symbol =
        context.find_symbol(data_object_symbol_type.identifier());
      assert(data_object_symbol);

      exprt new_base;
      get_ref_to_data_object(base, data_object_comp_or_vptr_comp, new_base);

      struct_typet data_object_type = to_struct_type(data_object_symbol->type);
      handle_components_of_data_object(
        ctor_type, ctor_body, data_object_type.components(), new_base);
    }
  }
}

void clang_cpp_adjust::gen_vptr_init_code(
  const struct_union_typet::componentt &vptr_comp,
  const exprt &base,
  side_effect_exprt &new_code,
  const code_typet &ctor_type)
{
  /*
   * Generate the statement to assign each vptr the corresponding
   * vtable address, e.g.:
   *  this->vptr = &<vtable_struct_var_name>
   */

  // 1. set the type
  //typet vtable_type = symbol_typet(vptr_comp.type().subtype().id());
  new_code.type() = vptr_comp.type();

  // 2. LHS: generate the member pointer dereference expression
  exprt lhs_expr = gen_vptr_init_lhs(vptr_comp, base, ctor_type);

  // 3. RHS: generate the address of the target virtual pointer struct
  exprt rhs_expr = gen_vptr_init_rhs(vptr_comp, ctor_type);

  // now push them to the assignment statement code
  new_code.operands().push_back(lhs_expr);
  new_code.operands().push_back(rhs_expr);
}

exprt clang_cpp_adjust::gen_vptr_init_lhs(
  const struct_union_typet::componentt &vptr_comp,
  const exprt &base,
  const code_typet &ctor_type)
{
  /*
   * Generate the LHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_variable>
   */

  exprt lhs_code;

  // now we can get the member expr for "base->vptr"
  assert(!base.type().is_pointer());
  lhs_code = member_exprt(base, vptr_comp.name(), vptr_comp.type());
  lhs_code.set("#lvalue", true);

  return lhs_code;
}

exprt clang_cpp_adjust::gen_vptr_init_rhs(
  const struct_union_typet::componentt &comp,
  const code_typet &ctor_type)
{
  /*
   * Generate the RHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_variable>
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
