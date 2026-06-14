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
  if (!symbol.get_value().need_vptr_init() || symbol.get_value().is_nil())
    return;

  /*
   * The "to_code_*" conversion functions have done some sanity checks for us:
   *  - a ctor symbol shall contain a function body modelled by code_blockt
   *  - symbol should be of code type
   */
  const code_typet &ctor_type = to_code_type(symbol.get_type());

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
    to_struct_type(ctor_class_symb->get_type()).components();

  // Read-modify-set the value to mutate the body and clear need_vptr_init.
  exprt value = symbol.get_value();
  code_blockt &ctor_body = to_code_block(to_code(value));

  // Initialize every vptr reachable from this class: the class' own vptr(s)
  // plus the vptr of each nested base subobject. Base subobjects are nested
  // members now (#is_base_subobject), so a base vptr lives at
  // `this->base::@base...->vptr`; we carry an access path so the LHS member
  // chain steps into the right subobject. RHS still points at the derived's
  // (thunk) vtable for that class, so dispatch through the base sees the
  // most-derived override.
  gen_vptr_inits_for_components(components, {}, ctor_body, ctor_type);

  value.need_vptr_init(false);
  symbol.set_value(std::move(value));
}

void clang_cpp_adjust::gen_vptr_inits_for_components(
  const struct_typet::componentst &components,
  const std::vector<struct_typet::componentt> &access_path,
  code_blockt &ctor_body,
  const code_typet &ctor_type)
{
  for (const auto &comp : components)
  {
    if (comp.get_bool("is_vtptr"))
    {
      side_effect_exprt new_code("assign");
      gen_vptr_init_code(comp, access_path, new_code, ctor_type);
      codet code_expr("expression");
      code_expr.move_to_operands(new_code);
      ctor_body.operands().push_back(code_expr);
    }
    else if (comp.get_bool("#is_base_subobject"))
    {
      // Recurse into the nested base subobject, extending the access path.
      const symbolt *base_symb =
        namespacet(context).lookup(comp.type().identifier());
      assert(base_symb);
      const struct_typet::componentst &base_components =
        to_struct_type(base_symb->get_type()).components();
      std::vector<struct_typet::componentt> nested = access_path;
      nested.push_back(comp);
      gen_vptr_inits_for_components(
        base_components, nested, ctor_body, ctor_type);
    }
  }
}

void clang_cpp_adjust::gen_vptr_init_code(
  const struct_union_typet::componentt &comp,
  const std::vector<struct_typet::componentt> &access_path,
  side_effect_exprt &new_code,
  const code_typet &ctor_type)
{
  /*
   * Generate the statement to assign each vptr the corresponding
   * vtable address, e.g.:
   *  this->vptr = &<vtable_struct_var_name>
   * For a base subobject vptr the LHS steps through the subobject chain in
   * access_path: this->base::@base...->vptr.
   */

  // 1. set the type
  //typet vtable_type = symbol_typet(comp.type().subtype().id());
  new_code.type() = comp.type();

  // 2. LHS: generate the member pointer dereference expression
  exprt lhs_expr = gen_vptr_init_lhs(comp, access_path, ctor_type);

  // 3. RHS: generate the address of the target virtual pointer struct
  exprt rhs_expr = gen_vptr_init_rhs(comp, access_path, ctor_type);

  // now push them to the assignment statement code
  new_code.operands().push_back(lhs_expr);
  new_code.operands().push_back(rhs_expr);
}

exprt clang_cpp_adjust::gen_vptr_init_lhs(
  const struct_union_typet::componentt &comp,
  const std::vector<struct_typet::componentt> &access_path,
  const code_typet &ctor_type)
{
  /*
   * Generate the LHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_variable>
   * or, for a base subobject vptr:
   *  this->base::@base...->vptr = &<vtable_struct_variable>
   */

  // get the `this` argument symbol
  const symbolt *this_symb = namespacet(context).lookup(
    ctor_type.arguments().at(0).type().subtype().identifier());
  assert(this_symb);

  // prepare dereference operand
  exprt deref_operand = symbol_exprt(
    ctor_type.arguments().at(0).get("#identifier"), this_symb->get_type());

  // get the reference symbol
  dereference_exprt this_deref(deref_operand.type());
  this_deref.operands().resize(0);
  this_deref.operands().push_back(deref_operand);

  // step into each nested base subobject along the access path
  exprt base_expr = this_deref;
  for (const auto &step : access_path)
  {
    exprt step_member = member_exprt(step.name(), step.type());
    step_member.operands().push_back(base_expr);
    base_expr = step_member;
  }

  // now we can get the member expr for "...->vptr"
  exprt lhs_code = member_exprt(comp.name(), comp.type());
  lhs_code.operands().push_back(base_expr);

  return lhs_code;
}

exprt clang_cpp_adjust::gen_vptr_init_rhs(
  const struct_union_typet::componentt &comp,
  const std::vector<struct_typet::componentt> &access_path,
  const code_typet &ctor_type)
{
  /*
   * Generate the RHS expression for virtual pointer initialization,
   * as in:
   *  this->vptr = &<vtable_struct_variable>
   */

  const std::string derived = ctor_type.get("#member_name").as_string();

  // Default: the vtable variable for this vptr's class in the most-derived
  // object (e.g. virtual_table::tag-Base@tag-Derived).
  std::string vtable_var_id =
    comp.type().subtype().identifier().as_string() + "@" + derived;

  // Itanium primary-base sharing: the vptr physically at offset 0 is the
  // primary base subobject's and carries the most-derived class' MERGED
  // vtable (primary view as a prefix + D's own). Detect that vptr — its access
  // path enters only the primary (first) base subobject at each level — and
  // retarget it to the merged vtable variable virtual_table::tag-D::merged@tag-D
  // that merge_primary_base_vtable built as the prefix-compatible superset.
  bool retargeted = false;
  if (is_primary_chain(derived, access_path))
  {
    const std::string merged_id =
      "virtual_table::" + derived + "::merged@" + derived;
    if (context.find_symbol(merged_id) != nullptr)
    {
      vtable_var_id = merged_id;
      retargeted = true;
    }
  }

  const symbolt *vtable_var_symb = namespacet(context).lookup(vtable_var_id);
  assert(vtable_var_symb);

  exprt vtable_var =
    symbol_exprt(vtable_var_symb->id, vtable_var_symb->get_type());
  vtable_var.name(vtable_var_symb->name);
  exprt addr = address_of_exprt(vtable_var);

  // The vptr slot is statically typed pointer(vtable_type::Base); the merged
  // table has its own (prefix-compatible) type, so cast the address to the
  // slot's pointer type. Reads index by name and stay valid via the prefix.
  if (retargeted)
    addr = typecast_exprt(addr, comp.type());
  return addr;
}

// Is access_path the primary (offset-0) base chain of class `derived` (its
// tag-prefixed id)? Such a vptr is the one physically shared at offset 0.
bool clang_cpp_adjust::is_primary_chain(
  const std::string &derived,
  const std::vector<struct_typet::componentt> &access_path)
{
  if (access_path.empty())
    return false;
  irep_idt cur = derived; // already tag-prefixed class id
  for (const auto &step : access_path)
  {
    const symbolt *s = context.find_symbol(cur);
    if (!s || s->get_type().id() != "struct")
      return false;
    irep_idt first_base;
    for (const auto &c : to_struct_type(s->get_type()).components())
      if (c.get_bool("#is_base_subobject"))
      {
        first_base = c.type().identifier();
        break;
      }
    if (step.type().identifier() != first_base)
      return false;
    cur = first_base;
  }
  return true;
}
