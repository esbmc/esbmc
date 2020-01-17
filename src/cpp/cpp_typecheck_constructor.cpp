/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_typecheck.h>
#include <cpp/cpp_util.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/std_code.h>
#include <util/std_expr.h>

static void copy_parent(
  const locationt &location,
  const irep_idt &parent_base_name,
  const irep_idt &arg_name,
  exprt &block)
{
  block.operands().push_back(codet());

  codet &code = to_code(block.operands().back());
  code.location() = location;

  code.set_statement("assign");
  code.operands().emplace_back("dereference");

  code.op0().operands().emplace_back("explicit-typecast");

  exprt &op0 = code.op0().op0();

  op0.operands().emplace_back("cpp-this");
  op0.type().id("pointer");
  op0.type().subtype().id("cpp-name");
  op0.type().subtype().get_sub().emplace_back("name");
  op0.type().subtype().get_sub().back().identifier(parent_base_name);
  op0.type().subtype().get_sub().back().set("#location", location);
  op0.location() = location;

  code.operands().emplace_back("explicit-typecast");
  exprt &op1 = code.op1();

  op1.type().id("pointer");
  op1.type().set("#reference", true);
  op1.type().subtype().set("#constant", true);
  op1.type().subtype().id("cpp-name");
  op1.type().subtype().get_sub().emplace_back("name");
  op1.type().subtype().get_sub().back().identifier(parent_base_name);
  op1.type().subtype().get_sub().back().set("#location", location);

  op1.operands().emplace_back("cpp-name");
  op1.op0().get_sub().emplace_back("name");
  op1.op0().get_sub().back().identifier(arg_name);
  op1.op0().get_sub().back().set("#location", location);
  op1.location() = location;
}

static void copy_member(
  const locationt &location,
  const irep_idt &member_base_name,
  const irep_idt &arg_name,
  exprt &block)
{
  block.operands().emplace_back("code");
  exprt &code = block.operands().back();

  code.statement("expression");
  code.type() = typet("code");
  code.operands().emplace_back("sideeffect");
  code.op0().statement("assign");
  code.op0().operands().emplace_back("cpp-name");
  code.location() = location;

  exprt &op0 = code.op0().op0();
  op0.location() = location;

  op0.get_sub().emplace_back("name");
  op0.get_sub().back().identifier(member_base_name);
  op0.get_sub().back().set("#location", location);

  code.op0().operands().emplace_back("member");

  exprt &op1 = code.op0().op1();

  op1.add("component_cpp_name").id("cpp-name");
  op1.add("component_cpp_name").get_sub().emplace_back("name");
  op1.add("component_cpp_name").get_sub().back().identifier(member_base_name);
  op1.add("component_cpp_name").get_sub().back().set("#location", location);

  op1.operands().emplace_back("cpp-name");
  op1.op0().get_sub().emplace_back("name");
  op1.op0().get_sub().back().identifier(arg_name);
  op1.op0().get_sub().back().set("#location", location);
  op1.location() = location;
}

static void copy_array(
  const locationt &location,
  const irep_idt &member_base_name,
  const BigInt &i,
  const irep_idt &arg_name,
  exprt &block)
{
  // Build the index expression
  exprt constant = from_integer(i, int_type());

  block.operands().emplace_back("code");
  exprt &code = block.operands().back();
  code.location() = location;

  code.statement("expression");
  code.type() = typet("code");
  code.operands().emplace_back("sideeffect");
  code.op0().statement("assign");
  code.op0().operands().emplace_back("index");
  exprt &op0 = code.op0().op0();
  op0.operands().emplace_back("cpp-name");
  op0.location() = location;

  op0.op0().get_sub().emplace_back("name");
  op0.op0().get_sub().back().identifier(member_base_name);
  op0.op0().get_sub().back().set("#location", location);
  op0.copy_to_operands(constant);

  code.op0().operands().emplace_back("index");

  exprt &op1 = code.op0().op1();
  op1.operands().emplace_back("member");
  op1.op0().add("component_cpp_name").id("cpp-name");
  op1.op0().add("component_cpp_name").get_sub().emplace_back("name");
  op1.op0()
    .add("component_cpp_name")
    .get_sub()
    .back()
    .identifier(member_base_name);
  op1.op0()
    .add("component_cpp_name")
    .get_sub()
    .back()
    .set("#location", location);

  op1.op0().operands().emplace_back("cpp-name");
  op1.op0().op0().get_sub().emplace_back("name");
  op1.op0().op0().get_sub().back().identifier(arg_name);
  op1.op0().op0().get_sub().back().set("#location", location);
  op1.copy_to_operands(constant);

  op1.location() = location;
}

void cpp_typecheckt::default_ctor(
  const locationt &location,
  const irep_idt &base_name,
  cpp_declarationt &ctor) const
{
  exprt name;
  name.id("name");
  name.identifier(base_name);
  name.location() = location;

  cpp_declaratort decl;
  decl.name().id("cpp-name");
  decl.name().move_to_sub(name);
  decl.type() = typet("function_type");
  decl.type().subtype().make_nil();
  decl.location() = location;

  decl.value().id("code");
  decl.value().type() = typet("code");
  decl.value().statement("block");
  decl.add("cv").make_nil();
  decl.add("throw_decl").make_nil();

  ctor.type().id("constructor");
  ctor.add("storage_spec").id("cpp-storage-spec");
  ctor.move_to_operands(decl);
  ctor.location() = location;
}

void cpp_typecheckt::default_cpctor(
  const symbolt &symbol,
  cpp_declarationt &cpctor) const
{
  locationt location = symbol.type.location();

  location.set_function(
    id2string(symbol.name) + "::" + id2string(symbol.name) + "( const " +
    id2string(symbol.name) + "&)");

  default_ctor(location, symbol.name, cpctor);
  cpp_declaratort &decl0 = cpctor.declarators()[0];

  std::string arg_name("ref");

  // Compound name
  irept compname("name");
  compname.identifier(symbol.name);
  compname.set("#location", location);

  cpp_namet cppcomp;
  cppcomp.move_to_sub(compname);

  // Argument name
  exprt argname("name");
  argname.location() = location;
  argname.identifier(arg_name);

  cpp_namet cpparg;
  cpparg.move_to_sub(argname);

  // Argument declarator
  cpp_declaratort argtor;
  argtor.add("value").make_nil();
  argtor.set("name", cpparg);
  argtor.type() = reference_typet();
  argtor.type().subtype().make_nil();
  argtor.type().add("#qualifier").make_nil();
  argtor.location() = location;

  // Argument declaration
  cpp_declarationt argdecl;
  argdecl.set("type", "merged_type");
  irept &subt = argdecl.type().add("subtypes");
  subt.get_sub().push_back(cppcomp);
  irept constnd("const");
  subt.get_sub().push_back(constnd);
  argdecl.move_to_operands(argtor);
  argdecl.location() = location;

  // Add argument to function type
  decl0.type().add("arguments").get_sub().push_back(argdecl);
  decl0.location() = location;

  irept &initializers = decl0.add("member_initializers");
  initializers.id("member_initializers");

  cpp_declaratort &declarator = (cpp_declaratort &)cpctor.op0();
  exprt &block = declarator.value();

  // First, we need to call the parent copy constructors
  const irept &bases = symbol.type.find("bases");
  forall_irep(parent_it, bases.get_sub())
  {
    assert(parent_it->id() == "base");
    assert(parent_it->get("type") == "symbol");

    const symbolt &parsymb = lookup(parent_it->type().identifier());

    if(cpp_is_pod(parsymb.type))
      copy_parent(location, parsymb.name, arg_name, block);
    else
    {
      irep_idt ctor_name = parsymb.name;

      // Call the parent default copy constructor
      exprt name("name");
      name.identifier(ctor_name);
      name.location() = location;

      cpp_namet cppname;
      cppname.move_to_sub(name);

      codet mem_init("member_initializer");
      mem_init.location() = location;
      mem_init.set("member", cppname);
      mem_init.add("operands").get_sub().push_back(cpparg);
      initializers.move_to_sub(mem_init);
    }
  }

  // Then, we add the member initializers
  const struct_typet::componentst &components =
    to_struct_type(symbol.type).components();
  for(struct_typet::componentst::const_iterator mem_it = components.begin();
      mem_it != components.end();
      mem_it++)
  {
    // Take care of virtual tables
    if(mem_it->get_bool("is_vtptr"))
    {
      exprt name("name");
      name.set("identifier", mem_it->base_name());
      name.location() = location;

      cpp_namet cppname;
      cppname.move_to_sub(name);

      const symbolt &virtual_table_symbol_type =
        namespacet(context).lookup(mem_it->type().subtype().identifier());

      const symbolt &virtual_table_symbol_var = namespacet(context).lookup(
        virtual_table_symbol_type.id.as_string() + "@" + symbol.id.as_string());

      exprt var = symbol_expr(virtual_table_symbol_var);
      address_of_exprt address(var);
      assert(address.type() == mem_it->type());

      already_typechecked(address);

      exprt ptrmember("ptrmember");
      ptrmember.set("component_name", mem_it->name());
      ptrmember.operands().emplace_back("cpp-this");

      code_assignt assign(ptrmember, address);
      initializers.move_to_sub(assign);
      continue;
    }

    if(
      mem_it->get_bool("from_base") || mem_it->is_type() ||
      mem_it->get_bool("is_static") || mem_it->type().id() == "code")
      continue;

    irep_idt mem_name = mem_it->base_name();

    exprt name("name");
    name.identifier(mem_name);
    name.location() = location;

    cpp_namet cppname;
    cppname.move_to_sub(name);

    codet mem_init("member_initializer");
    mem_init.set("member", cppname);
    mem_init.location() = location;

    exprt memberexpr("member");
    memberexpr.set("component_cpp_name", cppname);
    memberexpr.add("operands").get_sub().push_back(cpparg);
    memberexpr.location() = location;

    if(mem_it->type().id() == "array")
      memberexpr.set("#array_ini", true);

    mem_init.add("operands").get_sub().push_back(memberexpr);
    initializers.move_to_sub(mem_init);
  }

  cpctor.type().set("#default_copy_cons", "1");
}

void cpp_typecheckt::default_assignop(
  const symbolt &symbol,
  cpp_declarationt &cpctor)
{
  locationt location = symbol.type.location();

  location.set_function(
    id2string(symbol.name) + "& " + id2string(symbol.name) +
    "::operator=( const " + id2string(symbol.name) + "&)");

  std::string arg_name("ref");

  cpctor.add("storage_spec").id("cpp-storage-spec");
  cpctor.type().id("symbol");
  cpctor.type().add("identifier").id(symbol.id);
  cpctor.operands().emplace_back("cpp-declarator");
  cpctor.location() = location;

  cpp_declaratort &declarator = (cpp_declaratort &)cpctor.op0();
  declarator.location() = location;

  cpp_namet &declarator_name = declarator.name();
  typet &declarator_type = declarator.type();

  declarator_type.location() = location;

  declarator_name.id("cpp-name");
  declarator_name.get_sub().emplace_back("operator");
  declarator_name.get_sub().emplace_back("=");

  declarator_type.id("function_type");
  declarator_type.subtype() = reference_typet();
  declarator_type.subtype().add("#qualifier").make_nil();
  declarator_type.subtype().subtype().make_nil();

  exprt &args = (exprt &)declarator.type().add("arguments");
  args.location() = location;

  args.get_sub().emplace_back("cpp-declaration");

  cpp_declarationt &args_decl = (cpp_declarationt &)args.get_sub().back();

  irept &args_decl_type_sub = args_decl.type().add("subtypes");

  args_decl.type().id("merged_type");
  args_decl_type_sub.get_sub().emplace_back("cpp-name");
  args_decl_type_sub.get_sub().back().get_sub().emplace_back("name");
  args_decl_type_sub.get_sub().back().get_sub().back().identifier(symbol.name);
  args_decl_type_sub.get_sub().back().get_sub().back().set(
    "#location", location);

  args_decl_type_sub.get_sub().emplace_back("const");
  args_decl.operands().emplace_back("cpp-declarator");
  args_decl.location() = location;

  cpp_declaratort &args_decl_declor =
    (cpp_declaratort &)args_decl.operands().back();

  args_decl_declor.name().id("cpp-name");
  args_decl_declor.name().get_sub().emplace_back("name");
  args_decl_declor.name().get_sub().back().add("identifier").id(arg_name);
  args_decl_declor.location() = location;

  args_decl_declor.type().id("pointer");
  args_decl_declor.type().set("#reference", true);
  args_decl_declor.type().add("#qualifier").make_nil();
  args_decl_declor.type().subtype().make_nil();
  args_decl_declor.value().make_nil();
}

void cpp_typecheckt::default_assignop_value(
  const symbolt &symbol,
  cpp_declaratort &declarator)
{
  // save location
  locationt location = declarator.location();
  declarator.make_nil();

  declarator.value().location() = location;
  declarator.value().id("code");
  declarator.value().statement("block");
  declarator.value().type() = code_typet();

  exprt &block = declarator.value();

  std::string arg_name("ref");

  // First, we copy the parents
  const irept &bases = symbol.type.find("bases");

  forall_irep(parent_it, bases.get_sub())
  {
    assert(parent_it->id() == "base");
    assert(parent_it->get("type") == "symbol");

    const symbolt &symb = lookup(parent_it->type().identifier());

    copy_parent(location, symb.name, arg_name, block);
  }

  // Then, we copy the members
  const irept &components = symbol.type.components();

  forall_irep(mem_it, components.get_sub())
  {
    if(
      mem_it->get_bool("from_base") || mem_it->is_type() ||
      mem_it->get_bool("is_static") || mem_it->get_bool("is_vtptr") ||
      mem_it->get("type") == "code")
      continue;

    irep_idt mem_name = mem_it->base_name();

    if(mem_it->get("type") == "array")
    {
      const exprt &size_expr = to_array_type((typet &)mem_it->type()).size();

      if(size_expr.id() == "infinity")
      {
        // err_location(object);
        // err << "cannot copy array of infinite size" << std::endl;
        // throw 0;
        continue;
      }

      BigInt size;
      bool to_int = to_integer(size_expr, size);
      assert(!to_int);
      assert(size >= 0);
      (void)to_int; //ndebug

      exprt::operandst empty_operands;
      for(BigInt i = 0; i < size; ++i)
        copy_array(location, mem_name, i, arg_name, block);
    }
    else
      copy_member(location, mem_name, arg_name, block);
  }

  // Finally we add the return statement
  block.operands().emplace_back("code");
  exprt &ret_code = declarator.value().operands().back();
  ret_code.operands().emplace_back("dereference");
  ret_code.op0().operands().emplace_back("cpp-this");
  ret_code.statement("return");
  ret_code.type() = code_typet();
}

void cpp_typecheckt::check_member_initializers(
  const irept &bases,
  const struct_typet::componentst &components,
  const irept &initializers)
{
  assert(initializers.id() == "member_initializers");

  forall_irep(init_it, initializers.get_sub())
  {
    const irept &initializer = *init_it;
    assert(initializer.is_not_nil());

    std::string identifier, base_name;

    assert(initializer.get("member") == "cpp-name");

    const cpp_namet &member_name = to_cpp_name(initializer.member_irep());

    bool has_template_args = member_name.has_template_args();

    if(has_template_args)
    {
      // it has to be a parent constructor
      typet member_type = (typet &)initializer.member_irep();
      typecheck_type(member_type);

      // check for a direct parent
      bool ok = false;
      forall_irep(parent_it, bases.get_sub())
      {
        assert(parent_it->get("type") == "symbol");

        if(member_type.identifier() == parent_it->type().identifier())
        {
          ok = true;
          break;
        }
      }

      if(!ok)
      {
        err_location(member_name.location());
        str << "invalid initializer `" << member_name.to_string() << "'";
        throw 0;
      }
      return;
    }

    member_name.convert(identifier, base_name);
    bool ok = false;

    for(const auto &component : components)
    {
      if(component.base_name() != base_name)
        continue;

      // Data member
      if(
        !component.get_bool("from_base") && !component.get_bool("is_static") &&
        component.get("type") != "code")
      {
        ok = true;
        break;
      }

      // Maybe it is a parent constructor?
      if(component.is_type())
      {
        typet type = static_cast<const typet &>(component.type());
        if(type.id() != "symbol")
          continue;

        const symbolt &symb = lookup(type.identifier());
        if(symb.type.id() != "struct")
          break;

        // check for a direct parent
        forall_irep(parent_it, bases.get_sub())
        {
          assert(parent_it->get("type") == "symbol");
          if(symb.id == parent_it->type().identifier())
          {
            ok = true;
            break;
          }
        }
        continue;
      }

      // Parent constructor
      if(
        component.get_bool("from_base") && !component.is_type() &&
        !component.get_bool("is_static") && component.get("type") == "code" &&
        component.type().get("return_type") == "constructor")
      {
        typet member_type = (typet &)initializer.member_irep();
        typecheck_type(member_type);

        // check for a direct/indirect parent
        forall_irep(parent_it, bases.get_sub())
        {
          assert(parent_it->get("type") == "symbol");

          // check for a direct parent
          if(member_type.identifier() == parent_it->type().identifier())
          {
            ok = true;
            break;
          }

          // check for a indirect parent
          irep_idt identifier = parent_it->type().identifier();
          const symbolt &isymb = lookup(identifier);
          const typet &type = isymb.type;
          assert(type.id() == "struct");
          const irept &ibase = type.find("bases");
          forall_irep(iparent_it, ibase.get_sub())
          {
            assert(iparent_it->get("type") == "symbol");
            if(member_type.identifier() == iparent_it->type().identifier())
            {
              ok = true;
              break;
            }
          }
        }
        break;
      }
    }

    if(!ok)
    {
      std::cout << "ERROR " << std::endl;
      err_location(member_name.location());
      str << "invalid initializer `" << base_name << "'";
      throw 0;
    }
  }
}

void cpp_typecheckt::full_member_initialization(
  const struct_typet &struct_type,
  irept &initializers)
{
  const irept &bases = struct_type.find("bases");

  const struct_typet::componentst &components = struct_type.components();

  assert(initializers.id() == "member_initializers");

  irept final_initializers("member_initializers");

  // First, if we are the most-derived object, then
  // we need to construct the virtual bases
  std::list<irep_idt> vbases;
  get_virtual_bases(struct_type, vbases);

  if(!vbases.empty())
  {
    codet cond("ifthenelse");

    {
      cpp_namet most_derived;
      most_derived.get_sub().emplace_back("name");
      most_derived.get_sub().back().identifier("@most_derived");

      exprt tmp;
      tmp.swap(most_derived);
      cond.move_to_operands(tmp);
    }

    codet block("block");

    while(!vbases.empty())
    {
      const symbolt &symb = lookup(vbases.front());
      if(!cpp_is_pod(symb.type))
      {
        // default initializer
        irept name("name");
        name.identifier(symb.name);

        cpp_namet cppname;
        cppname.move_to_sub(name);

        codet mem_init("member_initializer");
        mem_init.set("member", cppname);
        block.move_to_sub(mem_init);
      }
      vbases.pop_front();
    }
    cond.move_to_operands(block);
    final_initializers.move_to_sub(cond);
  }

  // Subsequenlty, we need to call the non-POD parent constructors
  forall_irep(parent_it, bases.get_sub())
  {
    assert(parent_it->id() == "base");
    assert(parent_it->get("type") == "symbol");

    const symbolt &ctorsymb = lookup(parent_it->type().identifier());

    if(cpp_is_pod(ctorsymb.type))
      continue;

    irep_idt ctor_name = ctorsymb.name;

    // Check if the initialization list of the constructor
    // explicitly calls the parent constructor
    bool found = false;

    forall_irep(m_it, initializers.get_sub())
    {
      irept initializer = *m_it;
      assert(initializer.get("member") == "cpp-name");

      const cpp_namet &member_name = to_cpp_name(initializer.member_irep());

      bool has_template_args = member_name.has_template_args();

      if(!has_template_args)
      {
        std::string identifier;
        std::string base_name;
        member_name.convert(identifier, base_name);

        // check if the initializer is a data
        bool is_data = false;

        for(const auto &component : components)
        {
          if(
            component.base_name() == base_name &&
            component.get("type") != "code" && !component.is_type())
          {
            is_data = true;
            break;
          }
        }

        if(is_data)
          continue;
      }

      typet member_type = static_cast<const typet &>(initializer.member_irep());

      typecheck_type(member_type);

      if(member_type.id() != "symbol")
        break;

      if(parent_it->type().identifier() == member_type.identifier())
      {
        final_initializers.move_to_sub(initializer);
        found = true;
        break;
      }

      // initialize the indirect parent
      irep_idt identifier = parent_it->type().identifier();
      const symbolt &isymb = lookup(identifier);
      const typet &type = isymb.type;
      assert(type.id() == "struct");
      const irept &ibase = type.find("bases");
      forall_irep(iparent_it, ibase.get_sub())
      {
        assert(iparent_it->get("type") == "symbol");
        if(member_type.identifier() == iparent_it->type().identifier())
        {
          final_initializers.move_to_sub(initializer);
          found = true;
          break;
        }
      }
    }

    // Call the parent default constructor
    if(!found)
    {
      irept name("name");
      name.identifier(ctor_name);

      cpp_namet cppname;
      cppname.move_to_sub(name);

      codet mem_init("member_initializer");
      mem_init.set("member", cppname);
      final_initializers.move_to_sub(mem_init);
    }

    if(parent_it->get_bool("virtual"))
    {
      codet cond("ifthenelse");

      {
        cpp_namet most_derived;
        most_derived.get_sub().emplace_back("name");
        most_derived.get_sub().back().identifier("@most_derived");

        exprt tmp;
        tmp.swap(most_derived);
        cond.move_to_operands(tmp);
      }

      {
        codet tmp("member_initializer");
        tmp.swap(final_initializers.get_sub().back());
        cond.move_to_operands(tmp);
        final_initializers.get_sub().back().swap(cond);
      }
    }
  }

  // Then, we add the member initializers
  for(struct_typet::componentst::const_iterator mem_it = components.begin();
      mem_it != components.end();
      mem_it++)
  {
    // Take care of virtual tables
    if(mem_it->get_bool("is_vtptr"))
    {
      exprt name("name");
      name.set("identifier", mem_it->base_name());
      name.location() = mem_it->location();

      cpp_namet cppname;
      cppname.move_to_sub(name);

      const symbolt &virtual_table_symbol_type =
        lookup(mem_it->type().subtype().identifier());

      const symbolt &virtual_table_symbol_var = lookup(
        virtual_table_symbol_type.id.as_string() + "@" +
        struct_type.name().as_string());

      exprt var = symbol_expr(virtual_table_symbol_var);
      address_of_exprt address(var);
      assert(address.type() == mem_it->type());

      already_typechecked(address);

      exprt ptrmember("ptrmember");
      ptrmember.set("component_name", mem_it->name());
      ptrmember.operands().emplace_back("cpp-this");

      code_assignt assign(ptrmember, address);
      final_initializers.move_to_sub(assign);
      continue;
    }

    if(
      mem_it->get_bool("from_base") || mem_it->type().id() == "code" ||
      mem_it->is_type() || mem_it->get_bool("is_static"))
      continue;

    irep_idt mem_name = mem_it->base_name();

    // Check if the initialization list of the constructor
    // explicitly initializes the data member
    bool found = false;
    Forall_irep(m_it, initializers.get_sub())
    {
      irept &initializer = *m_it;
      std::string identifier;
      std::string base_name;

      if(initializer.get("member") != "cpp-name")
        continue;
      cpp_namet &member_name = (cpp_namet &)initializer.add("member");

      if(member_name.has_template_args())
        continue; // base-type initializer

      member_name.convert(identifier, base_name);

      if(mem_name == base_name)
      {
        final_initializers.move_to_sub(initializer);
        found = true;
        break;
      }
    }

    // If the data member is a reference, it must be explicitly
    // initialized
    if(!found && mem_it->type().id() == "pointer" && mem_it->type().reference())
    {
      err_location(*mem_it);
      str << "reference must be explicitly initialized";
      throw 0;
    }

    // If the data member is not POD and is not explicitly initialized,
    // then its default constructor is called.
    if(!found && !cpp_is_pod((const typet &)(mem_it->type())))
    {
      irept name("name");
      name.identifier(mem_name);

      cpp_namet cppname;
      cppname.move_to_sub(name);

      codet mem_init("member_initializer");
      mem_init.set("member", cppname);
      final_initializers.move_to_sub(mem_init);
    }
  }

  initializers.swap(final_initializers);
}

bool cpp_typecheckt::find_cpctor(const symbolt &symbol) const
{
  const struct_typet &struct_type = to_struct_type(symbol.type);
  const struct_typet::componentst &components = struct_type.components();

  for(const auto &component : components)
  {
    // Skip non-ctor
    if(
      component.type().id() != "code" ||
      to_code_type(component.type()).return_type().id() != "constructor")
      continue;

    // Skip inherited constructor
    if(component.get_bool("from_base"))
      continue;

    const code_typet &code_type = to_code_type(component.type());

    const code_typet::argumentst &args = code_type.arguments();

    // First argument is the this pointer. Therefore, copy
    // constructors have at least two arguments
    if(args.size() < 2)
      continue;

    const code_typet::argumentt &arg1 = args[1];

    const typet &arg1_type = arg1.type();

    if(!is_reference(arg1_type))
      continue;

    if(arg1_type.subtype().identifier() != symbol.id)
      continue;

    bool defargs = true;
    for(unsigned i = 2; i < args.size(); i++)
    {
      if(args[i].default_value().is_nil())
      {
        defargs = false;
        break;
      }
    }

    if(defargs)
      return true;
  }

  return false;
}

bool cpp_typecheckt::find_assignop(const symbolt &symbol) const
{
  const struct_typet &struct_type = to_struct_type(symbol.type);
  const struct_typet::componentst &components = struct_type.components();

  for(const auto &component : components)
  {
    if(component.base_name() != "operator=")
      continue;

    if(component.get_bool("is_static"))
      continue;

    if(component.get_bool("from_base"))
      continue;

    const code_typet &code_type = to_code_type(component.type());

    const code_typet::argumentst &args = code_type.arguments();

    if(args.size() != 2)
      continue;

    const code_typet::argumentt &arg1 = args[1];

    const typet &arg1_type = arg1.type();

    if(!is_reference(arg1_type))
      continue;

    if(arg1_type.subtype().identifier() != symbol.id)
      continue;

    return true;
  }

  return false;
}

bool cpp_typecheckt::find_dtor(const symbolt &symbol) const
{
  const irept &components = symbol.type.components();

  forall_irep(cit, components.get_sub())
  {
    if(cit->base_name() == "~" + id2string(symbol.name))
      return true;
  }

  return false;
}

void cpp_typecheckt::default_dtor(const symbolt &symb, cpp_declarationt &dtor)
{
  assert(symb.type.id() == "struct");

  irept name;
  name.id("name");
  name.identifier("~" + id2string(symb.name));
  name.set("#location", symb.location);

  cpp_declaratort decl;
  decl.name().id("cpp-name");
  decl.name().move_to_sub(name);
  decl.type().id("function_type");
  decl.type().subtype().make_nil();

  decl.value().id("code");
  decl.value().type().id("code");
  decl.value().statement("block");
  decl.add("cv").make_nil();
  decl.add("throw_decl").make_nil();

  dtor.type().id("destructor");
  dtor.add("storage_spec").id("cpp-storage-spec");
  dtor.add("operands").move_to_sub(decl);
}

void cpp_typecheckt::dtor(
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

      cpp_namet cppname;
      cppname.move_to_sub(name);

      const symbolt &virtual_table_symbol_type =
        namespacet(context).lookup(cit->type().subtype().identifier());

      const symbolt &virtual_table_symbol_var = namespacet(context).lookup(
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
    const symbolt &psymb = lookup(bit->type().identifier());

    exprt object("dereference");
    object.operands().emplace_back("cpp-this");
    object.location() = location;

    exprt dtor_code = cpp_destructor(location, psymb.type, object);

    if(dtor_code.is_not_nil())
      block.move_to_operands(dtor_code);
  }

  dtors = block;
}
