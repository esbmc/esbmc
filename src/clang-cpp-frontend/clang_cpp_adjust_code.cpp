#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <cpp/cpp_util.h>
#include <util/expr_util.h>

void convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

void clang_cpp_adjust::adjust_ifthenelse(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // if(void *p=...) ...
  if(code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_ifthenelse(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_ifthenelse(code);
}

void clang_cpp_adjust::adjust_while(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // while(void *p=...) ...
  if(code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_while(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_while(code);
}

void clang_cpp_adjust::adjust_switch(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // switch(int i=...) ...
  if(code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_switch(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_switch(code);
}

void clang_cpp_adjust::adjust_for(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // for( ; int i=...; ) ...
  if(code.op1().is_code())
  {
    codet decl_block = to_code(code.op1());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // Create new cond assignment
    code_declt &decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());
    assert(decl.operands().size() == 2);

    side_effect_exprt new_cond("assign", decl.op0().type());
    new_cond.copy_to_operands(decl.op0(), decl.op1());
    adjust_expr(new_cond);

    code.op1() = new_cond;
    clang_c_adjust::adjust_for(code);

    // Remove assignment
    decl.operands().pop_back();

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block, code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_for(code);
}

void clang_cpp_adjust::adjust_decl_block(codet &code)
{
  codet new_block("decl-block");

  Forall_operands(it, code)
  {
    if(it->is_code() && (it->statement() == "skip"))
      continue;

    code_declt &code_decl = to_code_decl(to_code(*it));

    if(code_decl.operands().size() == 2)
    {
      exprt &rhs = code_decl.rhs();
      exprt &lhs = code_decl.lhs();
      if(
        rhs.id() == "sideeffect" && rhs.statement() == "function_call" &&
        rhs.get_bool("constructor"))
      {
        // turn struct BLAH bleh = BLAH() into two instructions:
        // struct BLAH bleh;
        // BLAH(&bleh);

        // First, create new decl without rhs
        code_declt object(code_decl.lhs());
        new_block.copy_to_operands(object);

        // Get rhs
        side_effect_expr_function_callt &init =
          to_side_effect_expr_function_call(rhs);

        // Get lhs
        init.arguments().push_back(address_of_exprt(code_decl.lhs()));

        // Now convert the side_effect into an expression
        convert_expression_to_code(init);

        // and copy to new_block
        new_block.copy_to_operands(init);

        continue;
      }

      if(lhs.type().get_bool("#reference"))
      {
        // adjust rhs to address_of:
        // `int &r = g;` is turned into `int &r = &g;`
        exprt result_expr = exprt("address_of", rhs.type());
        result_expr.copy_to_operands(rhs.op0());
        rhs.swap(result_expr);
      }
    }

    new_block.copy_to_operands(code_decl);
  }

  code.swap(new_block);
}

void clang_cpp_adjust::adjust_code_block(codet &code)
{
  // if it is a destructor, add the implicit code
  if(code.get_bool("#is_dtor") && code.get_bool("#add_implicit_code"))
  {
    // get the correpsonding symbol using member_name
    const symbolt &msymb =
      *namespacet(context).lookup(code.get("#member_name"));

    // vtables should be updated as soon as the destructor is called
    // dtors contains the destructors for members and base classes,
    // that should be called after the code of the current destructor
    code_blockt vtables, dtors;
    get_vtables_dtors(msymb, vtables, dtors);

    if(vtables.has_operands())
      code.operands().insert(code.operands().begin(), vtables);

    if(dtors.has_operands())
      code.copy_to_operands(dtors);

    // now we have populated the code block for dtor
    // need to adjust the operands
    printf(
      "@@ adjust the operands? or continue adjust the code block for "
      "dtor???\n");
  }
}

void clang_cpp_adjust::get_vtables_dtors(
  const symbolt &symb,
  code_blockt &vtables,
  code_blockt &dtors)
{
  assert(symb.type.id() == "struct");

  locationt location = symb.type.location();

  location.set_function(
    id2string(symb.name) + "::~" + id2string(symb.name) + "()");

  const struct_typet::componentst &components =
    to_struct_type(symb.type).components();

  // take care of virtual methods
  for(struct_typet::componentst::const_iterator cit = components.begin();
      cit != components.end();
      cit++)
  {
    if(cit->get_bool("is_vtptr"))
    {
      exprt name("name");
      name.set("identifier", cit->base_name());

      const symbolt &virtual_table_symbol_type =
        *namespacet(context).lookup(cit->type().subtype().identifier());

      const symbolt &virtual_table_symbol_var = *namespacet(context).lookup(
        virtual_table_symbol_type.id.as_string() + "@" + symb.id.as_string());

      exprt var = symbol_expr(virtual_table_symbol_var);
      address_of_exprt address(var);
      assert(address.type() == cit->type());

      already_typechecked(address);

      exprt ptrmember("ptrmember");
      ptrmember.component_name(cit->name());
      ptrmember.operands().emplace_back("cpp-this");

      code_assignt assign(ptrmember, address);
      vtables.operands().push_back(assign);
      continue;
    }
  }

  code_blockt block;
#if 0
  // call the data member destructors in the reverse order
  for(struct_typet::componentst::const_reverse_iterator cit =
        components.rbegin();
      cit != components.rend();
      cit++)
  {
    const typet &type = cit->type();

    if(
      cit->get_bool("from_base") || cit->is_type() ||
      cit->get_bool("is_static") || type.id() == "code" || is_reference(type) ||
      cpp_is_pod(type))
      continue;

    irept name("name");
    name.identifier(cit->base_name());
    name.set("#location", location);

    cpp_namet cppname;
    cppname.get_sub().push_back(name);

    exprt member("ptrmember");
    member.set("component_cpp_name", cppname);
    member.operands().emplace_back("cpp-this");
    member.location() = location;

    codet dtor_code = cpp_destructor(location, cit->type(), member);

    if(dtor_code.is_not_nil())
      block.move_to_operands(dtor_code);
  }

  const irept::subt &bases = symb.type.find("bases").get_sub();

  // call the base destructors in the reverse order
  for(irept::subt::const_reverse_iterator bit = bases.rbegin();
      bit != bases.rend();
      bit++)
  {
    assert(bit->id() == "base");
    assert(bit->type().id() == "symbol");
    const symbolt &psymb = *lookup(bit->type().identifier());

    exprt object("dereference");
    object.operands().emplace_back("cpp-this");
    object.location() = location;

    exprt dtor_code = cpp_destructor(location, psymb.type, object);

    if(dtor_code.is_not_nil())
      block.move_to_operands(dtor_code);
  }
#endif

  dtors = block;
}
