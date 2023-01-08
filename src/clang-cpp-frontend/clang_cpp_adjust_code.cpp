#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <cpp/cpp_util.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <clang-c-frontend/typecast.h>

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
  //For class instantiation in C++, we need to adjust the side-effect of constructor
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
  if(code.get_bool("#is_dtor") && !code.get_bool("#added_implicit_code"))
  {
    // get the correpsonding symbol using member_name
    const symbolt &msymb =
      *namespacet(context).lookup(code.get("#member_name"));

    // vtables should be updated as soon as the destructor is called
    // dtors contains the destructors for members and base classes,
    // that should be called after the code of the current destructor
    code_blockt vtables, dtors;
    gen_vtables_dtors(msymb, vtables, dtors, code);

    // first we update the vtables, then we call the base destructors in reverse order
    if(vtables.has_operands())
      code.operands().insert(code.operands().begin(), vtables);

    if(dtors.has_operands())
      code.copy_to_operands(dtors);

    assert(code.statement() == "block"); // has to be a code block
    code.set("#added_implicit_code", true);
  }

  adjust_operands(code);
}

void clang_cpp_adjust::gen_vtables_dtors(
  const symbolt &symb,
  code_blockt &vtables,
  code_blockt &dtors,
  codet &code)
{
  // generate the implicit code for vtables and dtors
  assert(symb.type.id() == "struct");

  locationt location = symb.type.location();

  //location.set_function(id2string(symb.name) + "::~" + id2string(symb.name) + "()");

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
      exprt dtor_this("cpp-this");
      dtor_this.set("#this_arg", code.get("#this_arg"));
      ptrmember.operands().emplace_back(dtor_this);

      code_assignt assign(ptrmember, address);
      // special annotations for assigns in dtor implicit code
      assign.set("#dtor_implicit_code", true);
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
#endif

  const irept::subt &bases = symb.type.find("bases").get_sub();

  // call the base destructors in the reverse order
  for(irept::subt::const_reverse_iterator bit = bases.rbegin();
      bit != bases.rend();
      bit++)
  {
    assert(bit->id() == "base");
    assert(bit->type().id() == "symbol");
    const symbolt &psymb =
      *namespacet(context).lookup(bit->type().identifier());

    exprt dtor_code = gen_base_destructor(location, psymb.type, code);

    if(dtor_code.is_not_nil())
      block.move_to_operands(dtor_code);
  }

  dtors = block;
}

void clang_cpp_adjust::adjust_assign(codet &code)
{
  // Maps the flow in cpp_typecheckt::typecheck_assign

  // TODO: remove the condition and provide a unified way to adjust assign in C++
  if(code.get_bool("#dtor_implicit_code")) // add ctor/dtor implicit code
  {
    if(code.operands().size() != 2)
      throw "assignment statement expected to have two operands";

    // turn into a sideeffect
    side_effect_exprt expr(code.statement());
    expr.operands() = code.operands();
    // adjust operands in this expr, need to deal with ptrmember and already_typechecked
    adjust_expr(expr);

    code_expressiont code_expr;

    if(code_expr.operands().size() == 1)
    {
      // remove zombie operands
      if(code_expr.operands().front().id() == "")
        code_expr.operands().clear();
    }

    code_expr.copy_to_operands(expr);
    code_expr.location() = code.location();

    code.swap(code_expr);
  }
  else
  {
    // redirect everything else to clang-c's adjust_assign, same as before
    clang_c_adjust::adjust_assign(code);
  }
}

codet clang_cpp_adjust::gen_base_destructor(
  const locationt &location,
  const typet &type,
  codet &derived_dtor_code)
{
  // Maps the conversion flow to generate implicit destructor code
  //
  // Compared to the old frontend, this is a much simplified method to generate implicit
  // base dtor call.
  //
  // The old frontend uses cpp_typecheckt::cpp_destructor to generate an intermediate
  // irep, which is in turn "resolved" using the `cpp_typecheck_resolve` module to get the
  // base dtor function call. The `cpp_typecheck_resolve` module contains lots of code
  // to guess template args, exact type match, exact function match, identifier
  // disambuiation and some other "guessings" specific for the old typechecker.
  // I believe these "guessings" are already handled by the clang frontend converter.
  //
  // Since there cannot be more than one destructor in a class,
  // we just generate the base dtor call here.
  codet new_code;
  code_function_callt base_dtor_call;
  base_dtor_call.location() = location;

  typet tmp_type(type); // base class type
  namespacet(context).follow_symbol(tmp_type);

  assert(!is_reference(tmp_type));

  // PODs don't need a destructor
  if(cpp_is_pod(tmp_type))
  {
    base_dtor_call.make_nil();
    return base_dtor_call;
  }

  if(tmp_type.id() == "array")
  {
    log_error("TODO: Got array in {}", __func__);
    abort();
  }
  else
  {
    const struct_typet &struct_type = to_struct_type(tmp_type);

    irep_idt dtor_name;

    // get the components of the base class and search for dtor
    const struct_typet::componentst &components = struct_type.components();
    for(const auto &component : components)
    {
      const typet &type = component.type();

      if(
        !component.get_bool("from_base") && type.id() == "code" &&
        type.return_type().id() == "destructor")
      {
        // found base class dtor!
        dtor_name = component.base_name();

        // calling the base class dtor
        base_dtor_call.function() = symbol_exprt("symbol", component.type());
        base_dtor_call.function().identifier(component.name());
        ;

        // get the correct argument by type casting, something like:
        // ~BASE((BASE *)this);
        // where `this` represents the this operator in DERIVED dtor
        const code_typet &base_dtor_code_type = to_code_type(component.type());
        const code_typet::argumentst &base_dtor_arguments =
          base_dtor_code_type.arguments();
        // just one argument representing `this` in base class dtor
        assert(base_dtor_arguments.size() == 1);
        const typet base_dtor_arg_type = base_dtor_arguments.at(0).type();
        // generate the argument we want
        symbolt *s = context.find_symbol(derived_dtor_code.get("#this_arg"));
        const symbolt &this_symbol = *s;
        assert(s);
        exprt derived_dtor_this_symb = symbol_expr(this_symbol);
        gen_typecast(ns, derived_dtor_this_symb, base_dtor_arg_type);
        // use the argument in nase dtor function call
        base_dtor_call.arguments().push_back(derived_dtor_this_symb);

        // finally we set location of the base dtor function call
        base_dtor_call.location() = location;
        break;
      }
    }

    // there is always a destructor for non-PODs
    assert(dtor_name != "");
  }

  new_code.swap(base_dtor_call);
  return new_code;
}

bool clang_cpp_adjust::cpp_is_pod(const typet &type) const
{
  if(type.id() == "struct")
  {
    // Not allowed in PODs:
    // * Non-PODs
    // * Constructors/Destructors
    // * virtuals
    // * private/protected, unless static
    // * overloading assignment operator
    // * Base classes

    // XXX jmorse: certain things listed above don't always make their way into
    // the class definition though, such as templated constructors. In that
    // case, we set a flag to indicate that such methods have been seen, before
    // removing them. The "is_not_pod" flag thus only guarentees that it /isn't/
    // and its absence doesn't guarentee that it is.
    if(!type.find("is_not_pod").is_nil())
      return false;

    const struct_typet &struct_type = to_struct_type(type);

    if(!type.find("bases").get_sub().empty())
      return false;

    const struct_typet::componentst &components = struct_type.components();

    for(const auto &component : components)
    {
      if(component.is_type())
        continue;

      if(component.get_base_name() == "operator=")
        return false;

      if(component.get_bool("is_virtual"))
        return false;

      const typet &sub_type = component.type();

      if(sub_type.id() == "code")
      {
        if(component.get_bool("is_virtual"))
          return false;

        const typet &return_type = to_code_type(sub_type).return_type();

        if(
          return_type.id() == "constructor" || return_type.id() == "destructor")
          return false;
      }
      else if(
        component.get("access") != "public" && !component.get_bool("is_static"))
        return false;

      if(!cpp_is_pod(sub_type))
        return false;
    }

    return true;
  }
  if(type.id() == "array")
  {
    return cpp_is_pod(type.subtype());
  }
  else if(type.id() == "pointer")
  {
    if(is_reference(type)) // references are not PODs
      return false;

    // but pointers are PODs!
    return true;
  }
  else if(type.id() == "symbol")
  {
    const symbolt &symb = *namespacet(context).lookup(type.identifier());
    assert(symb.is_type);
    return cpp_is_pod(symb.type);
  }

  // everything else is POD
  return true;
}
