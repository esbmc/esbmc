#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <util/std_expr.h>

void clang_cpp_adjust::gen_vptr_initializations(symbolt &symbol)
{
  /*
   * This function traverses the vptr components of the correponding class type
   * in which the constructor is decleared. Then complete the initializations
   * for these pointers. The vptr is initialized like this:
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
   * The "to_code_*" conversion functions has done some sanity checks for us:
   *  - vptr initializations shall be done in ctor
   *  - a ctor symbol shall contain a function body modelled by code_blockt
   */
  code_typet &ctor_type = to_code_type(symbol.type);
  code_blockt &ctor_body = to_code_block(to_code(symbol.value));
  assert(ctor_type.return_type().id() == "constructor");

  // get the class' type where this ctor is declared
  const symbolt *ctor_class_symb =
    namespacet(context).lookup(ctor_type.get("#member_name"));
  assert(ctor_class_symb);
  // get the `components` vector from this class' type
  const struct_typet::componentst &components = to_struct_type(ctor_class_symb->type).components();

  printf("This is ctor function symbol: %s from class %s\n",
      symbol.id.as_string().c_str(),
      ctor_class_symb->id.as_string().c_str());

  // iterate over the `components` and initialize each virtual pointers
  for(const auto& comp : components)
  {
    if(!comp.get_bool("is_vtptr"))
      continue;

    side_effect_exprt new_code("assign");
    gen_vptr_init_code(comp, new_code, ctor_type);
    ctor_body.operands().push_back(new_code);
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
  exprt lhs_expr;
  gen_vptr_init_lhs(comp, lhs_expr, ctor_type);

  // 3. RHS: generate the address of the target virtual pointer struct
  exprt rhs_expr;
  gen_vptr_init_rhs(comp, rhs_expr, ctor_type);

  assert(!"Done - vptr init code");
}

void clang_cpp_adjust::gen_vptr_init_lhs(
  const struct_union_typet::componentt &comp,
  exprt &lhs_code,
  const code_typet &ctor_type)
{
  /*
   * Generate the LHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_var_name>
   */

  // get the `this` argument symbol
  const symbolt *this_symb =
    namespacet(context).lookup(
        ctor_type.arguments().at(0).type().subtype().identifier());
  assert(this_symb);

  // prepare dereference operand
  exprt deref_operand = symbol_exprt(
      ctor_type.arguments().at(0).get("#identifier"),
      pointer_typet(this_symb->type));

  // get the reference symbol
  dereference_exprt this_deref(deref_operand.type());
  this_deref.operands().resize(0);
  this_deref.operands().push_back(deref_operand);
  this_deref.set("#lvalue", true);

  // now we can get the member expr for "this->vptr"
  lhs_code = member_exprt(comp.name(), comp.type());
  lhs_code.operands().push_back(this_deref);
  lhs_code.set("#lvalue", true);
}

void clang_cpp_adjust::gen_vptr_init_rhs(
  const struct_union_typet::componentt &comp,
  exprt& rhs_code,
  const code_typet &ctor_type)
{
  /*
   * Generate the RHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_var_name>
   */

  assert(!"Done - addr of vtable");
}
