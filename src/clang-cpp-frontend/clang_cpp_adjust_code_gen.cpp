#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/std_expr.h>

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
  if(!symbol.value.need_vptr_init() || symbol.value.is_nil())
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
  if(ctor_type.return_type().id() != "constructor")
    return;

  // get the class' type where this ctor is declared
  const symbolt *ctor_class_symb =
    namespacet(context).lookup(ctor_type.get("#member_name"));
  assert(ctor_class_symb);
  // get the `components` vector from this class' type
  const struct_typet::componentst &components =
    to_struct_type(ctor_class_symb->type).components();

  // iterate over the `components` and initialize each virtual pointers
  for(const auto &comp : components)
  {
    if(!comp.get_bool("is_vtptr"))
      continue;

    side_effect_exprt new_code("assign");
    gen_vptr_init_code(comp, new_code, ctor_type);
    codet code_expr("expression");
    code_expr.move_to_operands(new_code);
    ctor_body.operands().push_back(code_expr);
  }

  symbol.value.need_vptr_init(false);
}

void clang_cpp_adjust::gen_vptr_init_code(
  const struct_union_typet::componentt &comp,
  side_effect_exprt &new_code,
  const code_typet &ctor_type)
{
  /*
   * Generate the statement to assign each vptr the corresponding
   * vtable address, e.g.:
   *  this->vptr = &<vtable_struct_var_name>
   */

  // 1. set the type
  //typet vtable_type = symbol_typet(comp.type().subtype().id());
  new_code.type() = comp.type();

  // 2. LHS: generate the member pointer dereference expression
  exprt lhs_expr = gen_vptr_init_lhs(comp, ctor_type);

  // 3. RHS: generate the address of the target virtual pointer struct
  exprt rhs_expr = gen_vptr_init_rhs(comp, ctor_type);

  // now push them to the assignment statement code
  new_code.operands().push_back(lhs_expr);
  new_code.operands().push_back(rhs_expr);
}

exprt clang_cpp_adjust::gen_vptr_init_lhs(
  const struct_union_typet::componentt &comp,
  const code_typet &ctor_type)
{
  /*
   * Generate the LHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_variable>
   */

  exprt lhs_code;

  // get the `this` argument symbol
  const symbolt *this_symb = namespacet(context).lookup(
    ctor_type.arguments().at(0).type().subtype().identifier());
  assert(this_symb);

  // prepare dereference operand
  exprt deref_operand = symbol_exprt(
    ctor_type.arguments().at(0).get("#identifier"), this_symb->type);

  // get the reference symbol
  dereference_exprt this_deref(deref_operand.type());
  this_deref.operands().resize(0);
  this_deref.operands().push_back(deref_operand);
  this_deref.set("#lvalue", true);

  // now we can get the member expr for "this->vptr"
  lhs_code = member_exprt(comp.name(), comp.type());
  lhs_code.operands().push_back(this_deref);
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
