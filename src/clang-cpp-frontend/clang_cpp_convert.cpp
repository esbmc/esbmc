// Remove warnings from Clang headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclFriend.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/QualTypeNames.h>
#include <clang/AST/Type.h>
#include <clang/Index/USRGeneration.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/AST/ParentMapContext.h>
#include <llvm/Support/raw_os_ostream.h>
#pragma GCC diagnostic pop

#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/i2string.h>
#include <fmt/core.h>
#include <cpp/cpp_util.h>
#include <clang-c-frontend/typecast.h>

clang_cpp_convertert::clang_cpp_convertert(
  contextt &_context,
  std::vector<std::unique_ptr<clang::ASTUnit>> &_ASTs,
  irep_idt _mode)
  : clang_c_convertert(_context, _ASTs, _mode),
    current_access(""),
    current_class_type(nullptr)
{
}

bool clang_cpp_convertert::get_decl(const clang::Decl &decl, exprt &new_expr)
{
  new_expr = code_skipt();

  switch(decl.getKind())
  {
  case clang::Decl::LinkageSpec:
  {
    const clang::LinkageSpecDecl &lsd =
      static_cast<const clang::LinkageSpecDecl &>(decl);

    for(auto decl : lsd.decls())
      if(get_decl(*decl, new_expr))
        return true;
    break;
  }

  case clang::Decl::CXXRecord:
  {
    const clang::CXXRecordDecl &cxxrd =
      static_cast<const clang::CXXRecordDecl &>(decl);

    // get class fields and methods
    if(get_struct_union_class(cxxrd))
      return true;

    break;
  }

  case clang::Decl::CXXConstructor:
  case clang::Decl::CXXMethod:
  case clang::Decl::CXXDestructor:
  case clang::Decl::CXXConversion:
  {
    const clang::CXXMethodDecl &cxxmd =
      static_cast<const clang::CXXMethodDecl &>(decl);

    assert(llvm::dyn_cast<clang::TemplateDecl>(&cxxmd) == nullptr);
    if(get_function(cxxmd, new_expr))
      return true;

    break;
  }

  case clang::Decl::Namespace:
  {
    const clang::NamespaceDecl &namesd =
      static_cast<const clang::NamespaceDecl &>(decl);

    for(auto decl : namesd.decls())
      if(get_decl(*decl, new_expr))
        return true;

    break;
  }

  case clang::Decl::FunctionTemplate:
  {
    const clang::FunctionTemplateDecl &fd =
      static_cast<const clang::FunctionTemplateDecl &>(decl);

    if(get_template_decl(&fd, true, new_expr))
      return true;
    break;
  }

  case clang::Decl::ClassTemplate:
  {
    const clang::ClassTemplateDecl &cd =
      static_cast<const clang::ClassTemplateDecl &>(decl);

    if(get_template_decl(&cd, false, new_expr))
      return true;
    break;
  }

  case clang::Decl::ClassTemplateSpecialization:
  {
    const clang::ClassTemplateSpecializationDecl &cd =
      static_cast<const clang::ClassTemplateSpecializationDecl &>(decl);

    if(get_struct_union_class(cd))
      return true;
    break;
  }

  case clang::Decl::Friend:
  {
    const clang::FriendDecl &fd = static_cast<const clang::FriendDecl &>(decl);

    if(fd.getFriendDecl() != nullptr)
      if(get_decl(*fd.getFriendDecl(), new_expr))
        return true;
    break;
  }

  case clang::Decl::AccessSpec:
  {
    const clang::AccessSpecDecl &asd =
      static_cast<const clang::AccessSpecDecl &>(decl);
    const clang::AccessSpecifier &specifier = asd.getAccess();
    current_access = get_access(specifier);
    break;
  }

  // We can ignore any these declarations
  case clang::Decl::ClassTemplatePartialSpecialization:
  case clang::Decl::Using:
  case clang::Decl::UsingShadow:
  case clang::Decl::UsingDirective:
  case clang::Decl::TypeAlias:
  case clang::Decl::NamespaceAlias:
  case clang::Decl::UnresolvedUsingValue:
  case clang::Decl::UnresolvedUsingTypename:
    break;

  default:
    return clang_c_convertert::get_decl(decl, new_expr);
  }

  return false;
}

bool clang_cpp_convertert::get_type(
  const clang::QualType &q_type,
  typet &new_type)
{
  return clang_c_convertert::get_type(q_type, new_type);
}

bool clang_cpp_convertert::get_type(
  const clang::Type &the_type,
  typet &new_type)
{
  switch(the_type.getTypeClass())
  {
  case clang::Type::SubstTemplateTypeParm:
  {
    const clang::SubstTemplateTypeParmType &substmpltt =
      static_cast<const clang::SubstTemplateTypeParmType &>(the_type);

    if(get_type(substmpltt.getReplacementType(), new_type))
      return true;
    break;
  }

  case clang::Type::TemplateSpecialization:
  {
    const clang::TemplateSpecializationType &templSpect =
      static_cast<const clang::TemplateSpecializationType &>(the_type);

    if(get_type(templSpect.desugar(), new_type))
      return true;
    break;
  }

  case clang::Type::MemberPointer:
  {
    const clang::MemberPointerType &mpt =
      static_cast<const clang::MemberPointerType &>(the_type);

    typet sub_type;
    if(get_type(mpt.getPointeeType(), sub_type))
      return true;

    typet class_type;
    if(get_type(*mpt.getClass(), class_type))
      return true;

    new_type = gen_pointer_type(sub_type);
    new_type.set("to-member", class_type);
    break;
  }

  default:
    return clang_c_convertert::get_type(the_type, new_type);
  }

  return false;
}

bool clang_cpp_convertert::get_var(const clang::VarDecl &vd, exprt &new_expr)
{
  // Only convert instantiated variables
  if(vd.getDeclContext()->isDependentContext())
    return false;

  return clang_c_convertert::get_var(vd, new_expr);
}

bool clang_cpp_convertert::get_function(
  const clang::FunctionDecl &fd,
  exprt &new_expr)
{
  // Only convert instantiated functions/methods not depending on a template parameter
  if(fd.isDependentContext())
    return false;

  return clang_c_convertert::get_function(fd, new_expr);
}

bool clang_cpp_convertert::get_struct_union_class(const clang::RecordDecl &rd)
{
  // Only convert RecordDecl not depending on a template parameter
  if(rd.isDependentContext())
    return false;

  return clang_c_convertert::get_struct_union_class(rd);
}

bool clang_cpp_convertert::get_struct_union_class_fields(
  const clang::RecordDecl &rd,
  struct_union_typet &type)
{
  return clang_c_convertert::get_struct_union_class_fields(rd, type);
}

bool clang_cpp_convertert::get_struct_union_class_methods(
  const clang::RecordDecl &recordd,
  struct_union_typet &type)
{
  // If a struct is defined inside a extern C, it will be a RecordDecl
  const clang::CXXRecordDecl *cxxrd =
    llvm::dyn_cast<clang::CXXRecordDecl>(&recordd);
  if(cxxrd == nullptr)
    return false;

  // default access: private for class, otherwise public
  current_access = type.get_bool("#class") ? "private" : "public";

  // control flags to:
  //  1. set is_not_pod
  //  2. manually make ctor/dtor
  //     (only in the old cpp frontend, clang automatically already adds
  //     `implicit` ctor, dtor, cpy ctor, cpy assignment optr in AST)
  bool found_ctor = false;
  bool found_dtor = false;

  current_class_type = &type;

  // Iterate over the declarations stored in this context
  // we first do everything but the constructors
  for(const auto &decl : cxxrd->decls())
  {
    // Fields were already added
    if(decl->getKind() == clang::Decl::Field)
      continue;

    // Clang AST contains an implicit CXXRecordDecl refering to the class itself
    // Skip it
    if(decl->getKind() == clang::Decl::CXXRecord && decl->isImplicit())
      continue;

    const auto fd = llvm::dyn_cast<clang::FunctionDecl>(decl);
    // Skip ctor. We need to do vtable before ctor
    if(fd)
    {
      if(fd->getKind() == clang::Decl::CXXConstructor)
      {
        found_ctor = true;
        continue;
      }

      if(fd->getKind() == clang::Decl::CXXDestructor)
        found_dtor = true;
    }

    const auto md = llvm::dyn_cast<clang::CXXMethodDecl>(decl);
    // Skip constructors only. We do implicit methods, such as implicit dtor,
    // implicit cpy ctor or cpy assignment operator
    // Add the default dtor, if needed
    // we have to do the destructor before building the virtual tables,
    // as the destructor may be virtual!
    if(md)
      if(fd->getKind() == clang::Decl::CXXConstructor)
        continue;

    struct_typet::componentt comp;
    if(
      const clang::FunctionTemplateDecl *ftd =
        llvm::dyn_cast<clang::FunctionTemplateDecl>(decl))
    {
      assert(ftd->isThisDeclarationADefinition());
      log_error("template is not supported in {}", __func__);
      abort();
    }
    else
    {
      if(get_decl(*decl, comp))
        return true;
    }

    // This means that we probably just parsed nested class or access specifier,
    // don't add it to the class
    if(comp.is_code() && to_code(comp).statement() == "skip")
    {
      // we need to add pure virtual method and destructor
      if(fd && md)
        if(
          !fd->isPure() && !md->isVirtual() &&
          fd->getKind() != clang::Decl::CXXDestructor)
          continue;
    }

    if(
      const clang::CXXMethodDecl *cxxmd =
        llvm::dyn_cast<clang::CXXMethodDecl>(decl))
    {
      // Add only if it isn't static
      // Both fields and methods are considered components in typechecking.
      // Methods will be moved from `components` to `methods` in adjuster.
      if(!cxxmd->isStatic())
        to_struct_type(type).components().push_back(comp);
      else
      {
        log_error("static method is not supported in {}", __func__);
        abort();
      }
    }
  }

  // If we've seen a constructor, flag this type as not being a POD. This is
  // only useful when we might not be able to work that out later, such as a
  // constructor that gets deleted, or something.
  if(found_ctor || found_dtor)
    type.set("is_not_pod", "1");

  // setup virtual tables before doing the constructors
  do_virtual_table(type);

  // reset current access before checking ctor
  current_access = type.get_bool("#class") ? "private" : "public";

  // All the data members are known now.
  // So let's deal with the constructors that we are given.
  for(const auto &decl : cxxrd->decls())
  {
    // Just do ctor in this iteration. Everything else should have been added.
    const auto fd = llvm::dyn_cast<clang::FunctionDecl>(decl);
    if(fd)
    {
      if(fd->getKind() != clang::Decl::CXXConstructor)
        continue;
    }
    else
    {
      // Skip non-method declarations except access specifier
      if(decl->getKind() != clang::Decl::AccessSpec)
        continue;
    }

    struct_typet::componentt comp;
    if(
      const clang::FunctionTemplateDecl *ftd =
        llvm::dyn_cast<clang::FunctionTemplateDecl>(decl))
    {
      assert(ftd->isThisDeclarationADefinition());
      log_error("template is not supported in {}", __func__);
      abort();
    }
    else
    {
      if(get_decl(*decl, comp))
        return true;
    }

    if(
      const clang::CXXMethodDecl *cxxmd =
        llvm::dyn_cast<clang::CXXMethodDecl>(decl))
    {
      // Add only if it isn't static
      if(!cxxmd->isStatic())
      {
        // Remove ctor's skip statement because we have vptr init.
        if(
          fd->getKind() == clang::Decl::CXXConstructor &&
          comp.statement() == "skip")
          comp.remove("statement");

        to_struct_type(type).components().push_back(comp);
      }
      else
      {
        log_error("static method is not supported in {}", __func__);
        abort();
      }
    }
  }

  // Restore access and class symbol type
  current_access = "";
  current_class_type = nullptr;

  return false;
}

bool clang_cpp_convertert::get_class_method(
  const clang::FunctionDecl &fd,
  exprt &component,
  code_typet &method_type,
  const symbolt &method_symbol)
{
  // Maps for method typechecking in cpp_typecheckt::typecheck_compound_declarator
  // This function typechecks C++ class methods:
  //  1. adding annotations to the `component` node in the parse tree (irep)
  //     The `component` refers to the irep node of a method within a class' symbol.type.components()
  //  2. performing additional typechecks for virtual method
  //  3. For constructors and destructors, the return_type will be `constructor` and `destructor` respectively
  if(const auto *md = llvm::dyn_cast<clang::CXXMethodDecl>(&fd))
  {
    assert(method_type.arguments()
             .begin()
             ->is_not_nil()); // "this" must have been added

    // setting ctor and dtor return type
    if(fd.getKind() == clang::Decl::CXXDestructor)
    {
      typet rtn_type("destructor");
      to_code_type(component.type()).return_type() = rtn_type;
    }
    if(fd.getKind() == clang::Decl::CXXConstructor)
    {
      typet rtn_type("constructor");
      to_code_type(component.type()).return_type() = rtn_type;
    }

    std::string class_symbol_id = method_type.arguments()
                                    .begin()
                                    ->type()
                                    .subtype()
                                    .identifier()
                                    .as_string();

    // For method defined outside the class declaration, i.e. not inlined
    // we need to figure out the current access specifier
    if(current_access == "")
      get_current_access(fd, *md->getParent());

    // Add the common annotations to all C++ class methods
    typet &component_type = component.type();
    component_type.set("#member_name", class_symbol_id);
    component.name(method_symbol.id);
    assert(current_access != "");
    component.set("access", current_access);
    component.base_name(method_symbol.name);
    component.pretty_name(method_symbol.name);
    component.set("is_inlined", fd.isInlined());
    component.location() = method_symbol.location;

    // Typechecking C++ virtual methods
    if(md->isVirtual())
      get_virtual_method(
        *md, component, method_type, method_symbol, class_symbol_id);

    // Sync virtual method's type with component's type after all checks done
    method_type = to_code_type(component_type);
  }
  else
  {
    log_error(
      "Checking cast to CXXMethodDecl failed in {}. Not a class method?",
      __func__);
    fd.dump();
    abort();
  }
  return false;
}

bool clang_cpp_convertert::get_virtual_method(
  const clang::FunctionDecl &fd,
  exprt &component,
  code_typet &method_type,
  const symbolt &method_symbol,
  const std::string &class_symbol_id)
{
  // Maps the virtual method typechecking in cpp_typecheckt::typecheck_compound_declarator
  // This function typechecks virtual methods,
  // adding annotations to the virtual method component irep tree
  // static non-method member are NOT handled here
  if(const auto *md = llvm::dyn_cast<clang::CXXMethodDecl>(&fd))
  {
    assert(md->isVirtual());
    assert(method_type.arguments()
             .begin()
             ->is_not_nil()); // "this" must have been added
    std::string virtual_name = method_symbol.name.as_string();
    std::string class_base_name = class_symbol_id;
    class_base_name.erase(0, tag_prefix.size());

    // Add additional annotations to C++ virtual method
    typet &component_type = component.type();
    component_type.set("#is_virtual", true);
    component_type.set("#virtual_name", virtual_name);
    component.set("is_pure_virtual", fd.isPure());
    component.set("virtual_name", virtual_name);
    component.set("is_virtual", true);

    // get the virtual-table symbol type
    irep_idt vt_name = "virtual_table::" + class_symbol_id;

    symbolt *s = context.find_symbol(vt_name);

    // add virtual_table symbol type
    if(s == nullptr)
    {
      // first time: create a virtual-table symbol type
      symbolt vt_symb_type;
      vt_symb_type.id = vt_name;
      vt_symb_type.name = "virtual_table::" + class_base_name;
      vt_symb_type.mode = mode;
      vt_symb_type.module = method_symbol.module;
      vt_symb_type.location = method_symbol.location;
      vt_symb_type.type = struct_typet();
      vt_symb_type.type.set("name", vt_symb_type.id);
      vt_symb_type.is_type = true;

      bool failed = context.move(vt_symb_type);
      assert(!failed);
      (void)failed; //ndebug

      s = context.find_symbol(vt_name);

      // add a virtual-table pointer
      struct_typet::componentt compo;
      compo.type() = pointer_typet(symbol_typet(vt_name));
      compo.set_name(class_symbol_id + "::@vtable_pointer");
      compo.base_name("@vtable_pointer");
      compo.pretty_name(class_base_name + "@vtable_pointer");
      compo.set("is_vtptr", true);
      compo.set("access", "public");
      assert(current_class_type);
      // add vptr component to class symbol type
      current_class_type->components().push_back(compo);
      // TODO: push_compound_into_scope?
    }
    else
    {
      log_error("Found existing vtable symbol type {}", __func__);
      abort();
    }

    assert(s->type.id() == "struct");

    struct_typet &virtual_table = to_struct_type(s->type);

    // add an entry to the virtual table
    // i.e. add virtual method component to virtual table symbol type
    struct_typet::componentt vt_entry;
    vt_entry.type() = pointer_typet(component_type);
    vt_entry.set_name(vt_name.as_string() + "::" + virtual_name);
    vt_entry.set("base_name", virtual_name);
    vt_entry.set("pretty_name", virtual_name);
    vt_entry.set("access", "public");
    vt_entry.location() = method_symbol.location;
    virtual_table.components().push_back(vt_entry);

    // take care of overloading:
    //  The bases should have been pulled in recursively.
    //  Walk through the components of method's parent class
    //  and add new function symbol for "late casting" of 'this' parameter
    std::set<irep_idt> virtual_bases;
    get_method_virtual_bases(virtual_bases, class_symbol_id, virtual_name);
    while(!virtual_bases.empty()) // TODO: move to another function?
    {
      irep_idt virtual_base = *virtual_bases.begin();

      // a new function that does 'late casting' of the 'this' parameter
      symbolt func_symb;
      func_symb.id =
        component.name().as_string() + "::" + virtual_base.as_string();
      func_symb.name = component.base_name();
      func_symb.mode = mode;
      func_symb.module =
        get_modulename_from_path(component.location().file().as_string());
      func_symb.location = component.location();
      func_symb.type = component.type();

      // change the type of the 'this' pointer
      code_typet &code_type = to_code_type(func_symb.type);
      code_typet::argumentt &arg = code_type.arguments().front();
      arg.type().subtype().set("identifier", virtual_base);

      // create symbols for the arguments
      code_typet::argumentst &args = code_type.arguments();
      for(unsigned i = 0; i < args.size(); i++)
      {
        code_typet::argumentt &arg = args[i];
        irep_idt base_name = arg.get_base_name();

        if(base_name == "")
          base_name = "arg" + i2string(i);

        symbolt arg_symb;
        arg_symb.id = func_symb.id.as_string() + "::" + base_name.as_string();
        arg_symb.name = base_name;
        arg_symb.mode = mode;
        arg_symb.location = func_symb.location;
        arg_symb.type = arg.type();

        arg.set("#identifier", arg_symb.id);

        // add the argument to the symbol table
        bool failed = context.move(arg_symb);
        assert(!failed);
        (void)failed; //ndebug
      }

      // do the body of the function
      typecast_exprt late_cast(
        to_code_type(component.type()).arguments()[0].type());

      late_cast.op0() =
        symbol_expr(*namespacet(context).lookup(args[0].cmt_identifier()));

      if(
        code_type.return_type().id() != "empty" &&
        code_type.return_type().id() != "destructor")
      {
        side_effect_expr_function_callt expr_call;
        expr_call.function() = symbol_exprt(component.name(), component.type());
        expr_call.type() = to_code_type(component.type()).return_type();
        expr_call.arguments().reserve(args.size());
        expr_call.arguments().push_back(late_cast);

        for(unsigned i = 1; i < args.size(); i++)
        {
          expr_call.arguments().push_back(
            symbol_expr(*namespacet(context).lookup(args[i].cmt_identifier())));
        }

        code_returnt code_return;
        code_return.return_value() = expr_call;

        func_symb.value = code_return;
      }
      else
      {
        log_error("Found destructor/empty return ids in {}", __func__);
        abort();
      }

      // add this new function to the list of components

      struct_typet::componentt new_compo(component);
      new_compo.type() = func_symb.type;
      new_compo.set_name(func_symb.id);
      current_class_type->components().push_back(new_compo);

      // add the function to the symbol table
      {
        bool failed = context.move(func_symb);
        assert(!failed);
        (void)failed; //ndebug
      }

      // next base
      virtual_bases.erase(virtual_bases.begin());
    }
  }
  else
  {
    log_error("Checking cast to CXXMethodDecl failed. Not a class method?");
    fd.dump();
    abort();
  }
  return false;
}

bool clang_cpp_convertert::get_expr(const clang::Stmt &stmt, exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(stmt, location);

  switch(stmt.getStmtClass())
  {
  case clang::Stmt::CXXReinterpretCastExprClass:
  // TODO: ReinterpretCast should actually generate a bitcast
  case clang::Stmt::CXXFunctionalCastExprClass:
  case clang::Stmt::CXXStaticCastExprClass:
  case clang::Stmt::CXXConstCastExprClass:
  {
    const clang::CastExpr &cast = static_cast<const clang::CastExpr &>(stmt);

    if(get_cast_expr(cast, new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXDefaultArgExprClass:
  {
    const clang::CXXDefaultArgExpr &cxxdarg =
      static_cast<const clang::CXXDefaultArgExpr &>(stmt);

    if(get_expr(*cxxdarg.getExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXDynamicCastExprClass:
  {
    const clang::CXXDynamicCastExpr &cast =
      static_cast<const clang::CXXDynamicCastExpr &>(stmt);

    if(cast.isAlwaysNull())
    {
      typet t;
      if(get_type(cast.getType(), t))
        return true;

      new_expr = gen_zero(gen_pointer_type(t));
    }
    else if(get_cast_expr(cast, new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXBoolLiteralExprClass:
  {
    const clang::CXXBoolLiteralExpr &bool_literal =
      static_cast<const clang::CXXBoolLiteralExpr &>(stmt);

    if(bool_literal.getValue())
      new_expr = true_exprt();
    else
      new_expr = false_exprt();
    break;
  }

  case clang::Stmt::CXXMemberCallExprClass:
  {
    const clang::CXXMemberCallExpr &member_call =
      static_cast<const clang::CXXMemberCallExpr &>(stmt);

    const clang::Stmt *callee = member_call.getCallee();

    exprt callee_expr;
    if(get_expr(*callee, callee_expr))
      return true;

    typet type;
    if(get_type(member_call.getType(), type))
      return true;

    side_effect_expr_function_callt call;
    call.function() = callee_expr;
    call.type() = type;

    // Add implicit object call: a this pointer or an object
    exprt implicit_object;
    if(get_expr(*member_call.getImplicitObjectArgument(), implicit_object))
      return true;

    call.arguments().push_back(implicit_object);

    // Do args
    for(const clang::Expr *arg : member_call.arguments())
    {
      exprt single_arg;
      if(get_expr(*arg, single_arg))
        return true;

      call.arguments().push_back(single_arg);
    }

    new_expr = call;
    break;
  }

  case clang::Stmt::CXXOperatorCallExprClass:
  {
    const clang::CXXOperatorCallExpr &operator_call =
      static_cast<const clang::CXXOperatorCallExpr &>(stmt);

    const clang::Stmt *callee = operator_call.getCallee();

    exprt callee_expr;
    if(get_expr(*callee, callee_expr))
      return true;

    typet type;
    if(get_type(operator_call.getType(), type))
      return true;

    side_effect_expr_function_callt call;
    call.function() = callee_expr;
    call.type() = type;

    // Do args
    for(const clang::Expr *arg : operator_call.arguments())
    {
      exprt single_arg;
      if(get_expr(*arg, single_arg))
        return true;

      call.arguments().push_back(single_arg);
    }

    new_expr = call;
    break;
  }

  case clang::Stmt::ExprWithCleanupsClass:
  {
    const clang::ExprWithCleanups &ewc =
      static_cast<const clang::ExprWithCleanups &>(stmt);

    if(get_expr(*ewc.getSubExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXBindTemporaryExprClass:
  {
    const clang::CXXBindTemporaryExpr &cxxbtmp =
      static_cast<const clang::CXXBindTemporaryExpr &>(stmt);

    if(get_expr(*cxxbtmp.getSubExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::SubstNonTypeTemplateParmExprClass:
  {
    const clang::SubstNonTypeTemplateParmExpr &substnttp =
      static_cast<const clang::SubstNonTypeTemplateParmExpr &>(stmt);

    if(get_expr(*substnttp.getReplacement(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::MaterializeTemporaryExprClass:
  {
    const clang::MaterializeTemporaryExpr &mtemp =
      static_cast<const clang::MaterializeTemporaryExpr &>(stmt);

    if(get_expr(*mtemp.getSubExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXNewExprClass:
  {
    const clang::CXXNewExpr &ne = static_cast<const clang::CXXNewExpr &>(stmt);

    typet t;
    if(get_type(ne.getType(), t))
      return true;

    if(ne.isArray())
    {
      new_expr = side_effect_exprt("cpp_new[]", t);

      // TODO: Implement support when the array size is empty
      assert(ne.getArraySize().hasValue());
      exprt size;
      if(get_expr(*(ne.getArraySize().getValue()), size))
        return true;

      new_expr.size(size);
    }
    else
    {
      new_expr = side_effect_exprt("cpp_new", t);
    }

    if(ne.hasInitializer())
    {
      exprt lhs("new_object", t);
      lhs.cmt_lvalue(true);

      exprt init;
      if(get_expr(*ne.getInitializer(), init))
        return true;

      convert_expression_to_code(init);

      new_expr.initializer(init);
    }

    break;
  }

  case clang::Stmt::CXXDeleteExprClass:
  {
    const clang::CXXDeleteExpr &de =
      static_cast<const clang::CXXDeleteExpr &>(stmt);

    new_expr = de.isArrayFormAsWritten()
                 ? side_effect_exprt("cpp_delete[]", empty_typet())
                 : side_effect_exprt("cpp_delete", empty_typet());

    exprt arg;
    if(get_expr(*de.getArgument(), arg))
      return true;

    new_expr.move_to_operands(arg);

    if(de.getDestroyedType()->getAsCXXRecordDecl())
    {
      typet destt;
      if(get_type(de.getDestroyedType(), destt))
        return true;
      new_expr.type() = destt;
    }

    break;
  }

  case clang::Stmt::CXXPseudoDestructorExprClass:
  {
    new_expr = exprt("pseudo_destructor");
    break;
  }

  case clang::Stmt::CXXScalarValueInitExprClass:
  {
    const clang::CXXScalarValueInitExpr &cxxsvi =
      static_cast<const clang::CXXScalarValueInitExpr &>(stmt);

    typet t;
    if(get_type(cxxsvi.getType(), t))
      return true;

    new_expr = gen_zero(t);
    break;
  }

  case clang::Stmt::TypeTraitExprClass:
  {
    const clang::TypeTraitExpr &tt =
      static_cast<const clang::TypeTraitExpr &>(stmt);

    if(tt.getValue())
      new_expr = true_exprt();
    else
      new_expr = false_exprt();
    break;
  }

  case clang::Stmt::CXXConstructExprClass:
  {
    const clang::CXXConstructExpr &cxxc =
      static_cast<const clang::CXXConstructExpr &>(stmt);

    // Avoid materializing a temporary for an elidable copy/move constructor.
    if(cxxc.isElidable() && !cxxc.requiresZeroInitialization())
    {
      const clang::MaterializeTemporaryExpr *mt =
        llvm::dyn_cast<clang::MaterializeTemporaryExpr>(cxxc.getArg(0));

      if(mt != nullptr)
      {
        log_error("elidable copy/move is not supported in {}", __func__);
        abort();
      }
    }

    if(get_constructor_call(cxxc, new_expr))
      return true;

    break;
  }

  case clang::Stmt::CXXThisExprClass:
  {
    const clang::CXXThisExpr &this_expr =
      static_cast<const clang::CXXThisExpr &>(stmt);

    std::size_t address =
      reinterpret_cast<std::size_t>(current_functionDecl->getFirstDecl());

    this_mapt::iterator it = this_map.find(address);
    if(this_map.find(address) == this_map.end())
    {
      log_error(
        "Pointer `this' for method {} was not added to scope",
        clang_c_convertert::get_decl_name(*current_functionDecl));
      abort();
    }

    typet this_type;
    if(get_type(this_expr.getType(), this_type))
      return true;

    assert(this_type == it->second.second);

    new_expr = symbol_exprt(it->second.first, it->second.second);
    break;
  }

  default:
    return clang_c_convertert::get_expr(stmt, new_expr);
  }

  new_expr.location() = location;
  return false;
}

bool clang_cpp_convertert::get_constructor_call(
  const clang::CXXConstructExpr &constructor_call,
  exprt &new_expr)
{
  // Get constructor call
  exprt callee_decl;
  if(get_decl_ref(*constructor_call.getConstructor(), callee_decl))
    return true;

  // Get type
  typet type;
  if(get_type(constructor_call.getType(), type))
    return true;

  side_effect_expr_function_callt call;
  call.function() = callee_decl;
  call.type() = type;

  // Try to get the object that this constructor is constructing
  auto it = ASTContext->getParents(constructor_call).begin();

  const clang::Decl *objectDecl = it->get<clang::Decl>();
  if(!objectDecl)
  {
    address_of_exprt tmp_expr;
    tmp_expr.type() = pointer_typet();
    tmp_expr.type().subtype() = type;

    exprt new_object("new_object");
    new_object.set("#lvalue", true);
    new_object.type() = type;

    tmp_expr.operands().resize(0);
    tmp_expr.move_to_operands(new_object);

    call.arguments().push_back(tmp_expr);
  }

  // Add implicit `this` before doing actual args
  // Given Motorcycle is a derived class from the base class Vehicle,
  // here the `#this_arg` represents `this` as in Motorcycle's ctor.
  // We need to wrap it in a typecast expr and convert to the Vehichle's `this`
  if(new_expr.get("#this_arg") != "")
  {
    // get base ctor this type
    const code_typet &base_ctor_code_type = to_code_type(callee_decl.type());
    const code_typet::argumentst &base_ctor_arguments =
      base_ctor_code_type.arguments();
    // just one argument representing `this` in base class ctor
    assert(base_ctor_arguments.size() == 1);
    const typet base_ctor_this_type = base_ctor_arguments.at(0).type();

    // get derived class ctor implicit this
    symbolt *s = context.find_symbol(new_expr.get("#this_arg"));
    const symbolt &this_symbol = *s;
    assert(s);
    exprt implicit_this_symb = symbol_expr(this_symbol);

    // generate the type casting expr and push it to callee's arguments
    gen_typecast(ns, implicit_this_symb, base_ctor_this_type);
    call.arguments().push_back(implicit_this_symb);
  }

  // Do args
  for(const clang::Expr *arg : constructor_call.arguments())
  {
    exprt single_arg;
    if(get_expr(*arg, single_arg))
      return true;

    call.arguments().push_back(single_arg);
  }

  call.set("constructor", 1);

  // We don't build a temporary obejct around it.
  // We follow the old cpp frontend to just add the ctor function call.
  new_expr.swap(call);

  return false;
}

void clang_cpp_convertert::build_member_from_component(
  const clang::FunctionDecl &fd,
  exprt &component)
{
  // Add this pointer as first argument
  std::size_t address = reinterpret_cast<std::size_t>(fd.getFirstDecl());

  this_mapt::iterator it = this_map.find(address);
  if(this_map.find(address) == this_map.end())
  {
    log_error(
      "Pointer `this' for method {} was not added to scope",
      clang_c_convertert::get_decl_name(fd));
    abort();
  }

  member_exprt member(
    symbol_exprt(it->second.first, it->second.second),
    component.name(),
    component.type());

  component.swap(member);
}

bool clang_cpp_convertert::get_function_body(
  const clang::FunctionDecl &fd,
  exprt &new_expr,
  const code_typet &ftype)
{
  // Constructor initializer list is checked here. Becasue we are going to convert
  // each initializer into an assignment statement, added to the function body.

  // Make a placeholder of code block. We add implicit code for dtor in the adjuster
  // TODO: refactor our ctor vptr initialization like this???
  if(is_dtor(fd))
  {
    new_expr = code_blockt();
    // Add additional annotations so that we can "catch" it in the adjuster
    // and add implicit code
    new_expr.set("#is_dtor", true);
    new_expr.set("#added_implicit_code", false);
    new_expr.set("#member_name", ftype.get("#member_name"));
    // this parameter must have been added
    assert(ftype.arguments().at(0).type().id() == "pointer");
    // get dtor `this` argument id for adjustment later
    new_expr.set("#this_arg", ftype.arguments().at(0).get("#identifier"));
    // sync dtor symbol.value.type with symbol.type
    new_expr.type() = ftype;
  }

  if(!fd.hasBody())
    return false;

  // Parse body
  if(clang_c_convertert::get_function_body(fd, new_expr, ftype))
    return true;

  code_blockt &body = to_code_block(to_code(new_expr));

  // if it's a constructor, check for initializers, e.g. initializers for data member, base class
  if(fd.getKind() == clang::Decl::CXXConstructor)
  {
    const clang::CXXConstructorDecl &cxxcd =
      static_cast<const clang::CXXConstructorDecl &>(fd);

    // Parse the initializers, if any
    if(cxxcd.init_begin() != cxxcd.init_end())
    {
      // Resize the number of operands
      exprt::operandst initializers;
      initializers.reserve(cxxcd.getNumCtorInitializers());

      for(auto init : cxxcd.inits())
      {
        exprt initializer;

        if(!init->isBaseInitializer())
        {
          exprt lhs;
          if(init->isMemberInitializer())
          {
            // parsing non-static member initializer
            if(get_decl_ref(*init->getMember(), lhs))
              return true;
          }
          else
          {
            log_error("Unsupported initializer in {}", __func__);
            abort();
          }

          build_member_from_component(fd, lhs);

          exprt rhs;
          if(get_expr(*init->getInit(), rhs))
            return true;

          initializer = side_effect_exprt("assign", lhs.type());
          initializer.copy_to_operands(lhs, rhs);
        }
        else
        {
          // Add additional annotation for `this` parameter
          initializer.set(
            "#this_arg", ftype.arguments().at(0).get("#identifier"));
          if(get_expr(*init->getInit(), initializer))
            return true;
        }

        // Convert to code and insert side-effect in the operands list
        // Essentially we convert an initializer to assignment, e.g:
        // t1() : i(2){ }
        // is converted to
        // t1() { this->i = 2; }
        convert_expression_to_code(initializer);
        initializers.push_back(initializer);
      }

      // Insert at the beginning of the body
      body.operands().insert(
        body.operands().begin(), initializers.begin(), initializers.end());
    }

    // TODO: move to adjuster! like what we did for dtor implicit code
    // Add implicit code for vptr initialization in ctor if needed.
    // Since clang AST does not contain virtual, we have to do the following things:
    //  - first we synthesize a `member_initializer` irep node for vptr initialization
    //  - then we convert the `member_initializer` node
    exprt vptr_init("code");
    if(get_vptr_init_irep(vptr_init, fd))
    {
      // sync value type with function type
      new_expr.type() = ftype;

      // populate ctor value using vptr_init's information
      get_vptr_init_expr(vptr_init, body, fd, ftype);
    }
  }

  return false;
}

bool clang_cpp_convertert::get_function_this_pointer_param(
  const clang::CXXMethodDecl &cxxmd,
  code_typet::argumentst &params)
{
  // Parse this pointer
  code_typet::argumentt this_param;
  if(get_type(cxxmd.getThisType(), this_param.type()))
    return true;

  locationt location_begin;
  get_location_from_decl(cxxmd, location_begin);

  std::string id, name;
  get_decl_name(cxxmd, name, id);

  name = "this";
  id += name;

  //this_param.cmt_base_name("this");
  this_param.cmt_base_name(name);
  this_param.cmt_identifier(id);

  // Add to the list of params
  params.push_back(this_param);

  // If the method is not defined, we don't need to add it's parameter
  // to the context, they will never be used
  if(!cxxmd.isDefined())
    return false;

  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    get_modulename_from_path(location_begin.file().as_string()),
    this_param.type(),
    name,
    id,
    location_begin);

  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  // Save the method address and name of this pointer on the this pointer map
  std::size_t address = reinterpret_cast<std::size_t>(cxxmd.getFirstDecl());
  this_map[address] = std::pair<std::string, typet>(
    param_symbol.id.as_string(), this_param.type());

  move_symbol_to_context(param_symbol);
  return false;
}

bool clang_cpp_convertert::get_function_params(
  const clang::FunctionDecl &fd,
  code_typet::argumentst &params)
{
  // On C++, all methods have an implicit reference to the
  // class of the object
  const clang::CXXMethodDecl &cxxmd =
    static_cast<const clang::CXXMethodDecl &>(fd);

  // If it's a C-style function, fallback to C mode
  // Static methods don't have the this arg and can be handled as
  // C functions
  if(!fd.isCXXClassMember() || cxxmd.isStatic())
    return clang_c_convertert::get_function_params(fd, params);

  // Add this pointer to first arg
  if(get_function_this_pointer_param(cxxmd, params))
    return true;

  // reserve space for `this' pointer and params
  params.resize(1 + fd.parameters().size());

  // Parse other args
  for(std::size_t i = 0; i < fd.parameters().size(); ++i)
  {
    code_typet::argumentt param;
    if(get_function_param(*fd.parameters()[i], param))
      return true;

    // All args are added shifted by one position, because
    // of the this pointer (first arg)
    params[i + 1].swap(param);
  }

  return false;
}

template <typename SpecializationDecl>
bool clang_cpp_convertert::get_template_decl_specialization(
  const SpecializationDecl *D,
  bool DumpExplicitInst,
  bool,
  exprt &new_expr)
{
  for(auto *redecl_with_bad_type : D->redecls())
  {
    auto *redecl = llvm::dyn_cast<SpecializationDecl>(redecl_with_bad_type);
    if(!redecl)
    {
      assert(
        llvm::isa<clang::CXXRecordDecl>(redecl_with_bad_type) &&
        "expected an injected-class-name");
      continue;
    }

    switch(redecl->getTemplateSpecializationKind())
    {
    case clang::TSK_ExplicitInstantiationDeclaration:
    case clang::TSK_ExplicitInstantiationDefinition:
    case clang::TSK_ExplicitSpecialization:
      if(!DumpExplicitInst)
        break;
      // Fall through.
    case clang::TSK_Undeclared:
    case clang::TSK_ImplicitInstantiation:
      if(get_decl(*redecl, new_expr))
        return true;
      break;
    }
  }

  return false;
}

template <typename TemplateDecl>
bool clang_cpp_convertert::get_template_decl(
  const TemplateDecl *D,
  bool DumpExplicitInst,
  exprt &new_expr)
{
  for(auto *Child : D->specializations())
    if(get_template_decl_specialization(
         Child, DumpExplicitInst, !D->isCanonicalDecl(), new_expr))
      return true;

  return false;
}

bool clang_cpp_convertert::get_decl_ref(
  const clang::Decl &decl,
  exprt &new_expr)
{
  std::string name, id;
  typet type;

  switch(decl.getKind())
  {
  case clang::Decl::Var:
  case clang::Decl::Field:
  {
    return clang_c_convertert::get_decl_ref(decl, new_expr);
  }
  case clang::Decl::CXXConstructor:
  {
    const clang::FunctionDecl &fd =
      static_cast<const clang::FunctionDecl &>(decl);

    get_decl_name(fd, name, id);

    if(get_type(fd.getType(), type))
      return true;

    if(get_function_params(fd, to_code_type(type).arguments()))
      return true;

    break;
  }

  default:
  {
    // Cases not handled above are unknown clang decls; we print an warning.
    // It might be possible to support them either here or in clang_c_frontend::get_decl_ref()
    // depending on whether they are C++-specific or not.
    std::ostringstream oss;
    llvm::raw_os_ostream ross(oss);
    decl.dump(ross);
    ross.flush();
    log_warning(
      "Conversion of unsupported clang decl ref for: {}\n{}",
      decl.getDeclKindName(),
      oss.str());
    return true;
  }
  }

  new_expr = exprt("symbol", type);
  new_expr.identifier(id);
  new_expr.cmt_lvalue(true);
  new_expr.name(name);

  return false;
}

std::string
clang_cpp_convertert::get_access(const clang::AccessSpecifier &specifier)
{
  if(specifier == clang::AS_public)
    return "public";
  else if(specifier == clang::AS_private)
    return "private";
  else if(specifier == clang::AS_protected)
    return "protected";
  else
  {
    log_error("Unknown accessor specified returned from clang in {}", __func__);
    abort();
  }
}

void clang_cpp_convertert::do_virtual_table(const struct_union_typet &type)
{
  // builds virtual-table value maps: (class x virtual_name x value)
  std::map<irep_idt, std::map<irep_idt, exprt>> vt_value_maps;

  const struct_typet &struct_type = to_struct_type(type);
  for(const auto &compo : struct_type.components())
  {
    if(!compo.get_bool("is_virtual"))
      continue;

    const code_typet &code_type = to_code_type(compo.type());
    assert(code_type.arguments().size() > 0);

    const pointer_typet &pointer_type =
      static_cast<const pointer_typet &>(code_type.arguments()[0].type());

    irep_idt class_id = pointer_type.subtype().identifier();

    std::map<irep_idt, exprt> &value_map = vt_value_maps[class_id];

    exprt e = symbol_exprt(compo.get_name(), code_type);

    if(compo.get_bool("is_pure_virtual"))
    {
      pointer_typet pointer_type(code_type);
      e = gen_zero(pointer_type);
      assert(e.is_not_nil());
      value_map[compo.get("virtual_name")] = e;
    }
    else
    {
      address_of_exprt address(e);
      value_map[compo.get("virtual_name")] = address;
    }
  }

  // create virtual-table symbol variables
  for(std::map<irep_idt, std::map<irep_idt, exprt>>::const_iterator cit =
        vt_value_maps.begin();
      cit != vt_value_maps.end();
      cit++)
  {
    const std::map<irep_idt, exprt> &value_map = cit->second;

    const symbolt &late_cast_symb = *namespacet(context).lookup(cit->first);
    const symbolt &vt_symb_type = *namespacet(context).lookup(
      "virtual_table::" + late_cast_symb.id.as_string());

    symbolt vt_symb_var;
    vt_symb_var.id =
      vt_symb_type.id.as_string() + "@" + type.name().as_string();
    vt_symb_var.name =
      vt_symb_type.name.as_string() + "@" + type.tag().as_string();
    vt_symb_var.mode = mode;
    vt_symb_var.module =
      get_modulename_from_path(type.location().file().as_string());
    vt_symb_var.location = vt_symb_type.location;
    vt_symb_var.type = symbol_typet(vt_symb_type.id);
    vt_symb_var.lvalue = true;
    vt_symb_var.static_lifetime = true;

    // do the values
    const struct_typet &vt_type = to_struct_type(vt_symb_type.type);
    exprt values("struct", symbol_typet(vt_symb_type.id));
    for(const auto &compo : vt_type.components())
    {
      std::map<irep_idt, exprt>::const_iterator cit2 =
        value_map.find(compo.base_name());
      assert(cit2 != value_map.end());
      const exprt &value = cit2->second;
      assert(value.type() == compo.type());
      values.operands().push_back(value);
    }
    vt_symb_var.value = values;

    bool failed = context.move(vt_symb_var);
    assert(!failed);
    (void)failed; // ndebug
  }
}

bool clang_cpp_convertert::get_vptr_init_irep(
  exprt &vptr_init,
  const clang::FunctionDecl &fd)
{
  // search for vptr component in current class symbol type
  // and make an assign statement to represent:
  //    this->vptr = &vtable;

  // vptr initialization comes from a constructor's body.
  // if we are converting a constructor body defined outside the class,
  // we need to retrieve the class symbol type
  if(current_class_type == nullptr)
    current_class_type =
      &to_struct_union_type(get_parent_class_symbol(fd)->type);

  bool found_vptr = false;
  const struct_union_typet::componentst &components =
    current_class_type->components();

  for(struct_union_typet::componentst::const_iterator mem_it =
        components.begin();
      mem_it != components.end();
      mem_it++)
  {
    if(mem_it->get_bool("is_vtptr"))
    {
      found_vptr = true;
      const symbolt &virtual_table_symbol_type =
        *context.find_symbol(mem_it->type().subtype().identifier());

      const symbolt &virtual_table_symbol_var = *context.find_symbol(
        virtual_table_symbol_type.id.as_string() + "@" +
        current_class_type->name().as_string());

      exprt var = symbol_expr(virtual_table_symbol_var);
      address_of_exprt address(var);
      assert(address.type() == mem_it->type());

      already_typechecked(address);

      exprt ptrmember("ptrmember");
      ptrmember.set("component_name", mem_it->name());
      ptrmember.operands().emplace_back("cpp-this");

      code_assignt assign(ptrmember, address);
      vptr_init.move_to_operands(assign);
    }
  }

  return found_vptr;
}

void clang_cpp_convertert::get_vptr_init_expr(
  const exprt &vptr_init,
  code_blockt &body,
  const clang::FunctionDecl &fd,
  const code_typet &ftype)
{
  // This function populates ctor symbol value for vptr initialization
  // if we are converting a constructor, we are in the middle of converting a class
  assert(current_class_type);

  // now we populate function body using vptr_init information
  codet vptr_init_code = to_code(vptr_init.operands()[0]);

  // it has to be an "assign" statement to represent:
  //    this->vptr = &vtable;
  assert(vptr_init_code.statement() == "assign");

  codet vptr_assign("expression");
  vptr_assign.type() = code_typet();

  // now let's do side effect for vptr init
  side_effect_exprt expr(vptr_init_code.statement());
  expr.set("#lvalue", true);

  auto lhs = vptr_init_code.operands()[0];
  auto rhs = vptr_init_code.operands()[1].operands().at(0);
  // rhs has to be the address of vtable, i.e. "&vtable".
  assert(rhs.id() == "address_of");
  // type should be rhs type
  expr.type() = rhs.type();

  // do lhs: this->vptr
  exprt member("member", rhs.type());
  member.set("component_name", lhs.component_name());
  member.set("#lvalue", true);
  // get "this" pointer id from function declaration
  const clang::CXXMethodDecl &cxxmd =
    static_cast<const clang::CXXMethodDecl &>(fd);
  std::string id, name;
  get_decl_name(cxxmd, name, id);
  name = "this";
  id += name;

  auto this_type = ftype.arguments().at(0).type();
  exprt deref("dereference", this_type.subtype());
  deref.set("#lvalue", true);

  const symbolt *ctor_this_symbol = context.find_symbol(id);
  exprt tmp("symbol", ctor_this_symbol->type);
  tmp.identifier(ctor_this_symbol->id);
  if(ctor_this_symbol->lvalue)
    tmp.set("#lvalue", true);
  deref.operands().push_back(tmp);

  member.operands().push_back(deref);
  expr.operands().push_back(member);

  // do rhs: &vtable
  expr.operands().push_back(rhs);

  // add side effect to ctor symbol value's operands
  vptr_assign.operands().push_back(expr);
  body.operands().push_back(vptr_assign);
}

void clang_cpp_convertert::get_current_access(
  const clang::FunctionDecl &target_fd,
  const clang::CXXRecordDecl &cxxrd)
{
  // default access
  current_access = cxxrd.getDefinition()->isClass() ? "private" : "public";

  for(const auto &decl : cxxrd.decls())
  {
    const auto asd = llvm::dyn_cast<clang::AccessSpecDecl>(decl);
    const auto current_fd = llvm::dyn_cast<clang::FunctionDecl>(decl);

    // set current access
    if(asd)
    {
      exprt dummy;
      get_decl(*decl, dummy);
    }

    // stop when reaching target function declaration
    if(current_fd)
    {
      std::string target_fd_id, current_fd_id;
      std::string dummy_name;
      get_decl_name(target_fd, dummy_name, target_fd_id);
      get_decl_name(*current_fd, dummy_name, current_fd_id);

      if(target_fd_id == current_fd_id)
        break;
    }
  }
}

symbolt *clang_cpp_convertert::get_parent_class_symbol(
  const clang::FunctionDecl &target_fd)
{
  // This is NOT the parent class to a child class.
  // It's the parent class for non-inlined methods.
  symbolt *s = nullptr;
  if(const auto *md = llvm::dyn_cast<clang::CXXMethodDecl>(&target_fd))
  {
    const clang::CXXRecordDecl *parent_rd = md->getParent();
    assert(parent_rd);

    std::string id, name;
    get_decl_name(*parent_rd, name, id);

    s = context.find_symbol(id);
  }
  else
  {
    log_error(
      "Checking cast to CXXMethodDecl failed in {}. Not a class method?",
      __func__);
    target_fd.dump();
    abort();
  }

  assert(s);
  return s;
}

bool clang_cpp_convertert::get_bases(
  const clang::RecordDecl &rd,
  struct_typet &derived_class_type)
{
  // copy components from base(s) into this class

  const clang::CXXRecordDecl &cxxrd =
    static_cast<const clang::CXXRecordDecl &>(rd);

  // skip incomplete class
  if(!cxxrd.isCompleteDefinition())
    return false;

  // skip if this class does not have any base
  if(cxxrd.bases().empty())
    return false;

  get_irep_base_vector(cxxrd, derived_class_type);

  std::set<irep_idt> bases;
  std::set<irep_idt> vbases;

  for(const clang::CXXBaseSpecifier &decl : cxxrd.bases())
  {
    // The base class is always a CXXRecordDecl
    const clang::CXXRecordDecl &base_cxxrd =
      *(decl.getType().getTypePtr()->getAsCXXRecordDecl());

    // get base class id
    std::string base_id, base_name;
    get_decl_name(base_cxxrd, base_name, base_id);

    // get base class symbol
    const symbolt &base_symbol = *namespacet(context).lookup(base_id);
    assert(base_symbol.type.id() == "struct");
    const struct_typet &base_struct_type = to_struct_type(base_symbol.type);

    bool is_virtual = decl.isVirtual(); // virtual base
    clang::AccessSpecifier class_access =
      decl.getAccessSpecifier(); // inheritance access specifier

    // TODO: added derived_class_type.add("bases") ? See base_symbol_expr in cpp_typecheckt::typecheck_compound_bases

    add_base_components(
      base_cxxrd,
      base_struct_type,
      class_access,
      derived_class_type,
      bases,
      vbases,
      is_virtual);
  }

  if(!vbases.empty())
  {
    log_error(
      "Got non-empty vbases in {}. Need most_derived component?", __func__);
    abort();
  }

  // DEBUG - print
  printf("@@ Done base parsing\n");
  printf(
    "@@ Bases for class: %s\n", derived_class_type.tag().as_string().c_str());
  printf("- printing bases: ");
  for(const auto &base : bases)
  {
    printf("%s ", base.as_string().c_str());
  }
  printf("\n");
  printf("- printing vbases: ");
  for(const auto &vbase : vbases)
  {
    printf("%s ", vbase.as_string().c_str());
  }
  printf("\n");
  printf("@@ Done pulling bases in ...\n");

  return false;
}

void clang_cpp_convertert::add_base_components(
  const clang::CXXRecordDecl &base_cxxrd,
  const struct_typet &from,
  const clang::AccessSpecifier &access,
  struct_typet &to,
  std::set<irep_idt> &bases,
  std::set<irep_idt> &vbases,
  bool is_virtual)
{
  const irep_idt &from_name = from.name();

  if(
    is_virtual && vbases.find(from_name) !=
                    vbases.end()) // nothing to add if virtual inheritance
    return;

  // Check for multiple inheritance of non-virtual base class
  // Clang should've reported parsing error
  assert(bases.find(from_name) == bases.end());

  // update bases and vbases
  bases.insert(from_name);
  if(is_virtual)
    vbases.insert(from_name);

  // look at the inheritance hierarchy of the base class
  for(const clang::CXXBaseSpecifier &decl : base_cxxrd.bases())
  {
    // The base class is always a CXXRecordDecl
    const clang::CXXRecordDecl &base =
      *(decl.getType().getTypePtr()->getAsCXXRecordDecl());

    // work out access specifier
    clang::AccessSpecifier sub_access = clang::AS_none;
    if(access == clang::AS_private)
      sub_access = clang::AS_private;
    else if(access == clang::AS_protected)
      sub_access = clang::AS_protected;
    else
      sub_access = clang::AS_public;

    // get class id
    std::string base_id, base_name;
    get_decl_name(base_cxxrd, base_name, base_id);

    // get base class symbol
    const symbolt &symb = *namespacet(context).lookup(base_id);
    assert(symb.type.id() == "struct");
    bool is_virtual = decl.isVirtual(); // virtual base

    // recursive call
    add_base_components(
      base,
      to_struct_type(symb.type),
      sub_access,
      to,
      bases,
      vbases,
      is_virtual);
  }

  // add the components
  const struct_typet::componentst &src_c = from.components();
  struct_typet::componentst &dest_c = to.components();

  for(const auto &it : src_c)
  {
    if(it.get_bool("from_base"))
      continue;

    // copy the component
    dest_c.push_back(it);

    // now twiddle the copy
    struct_typet::componentt &component = dest_c.back();
    component.set("from_base", true);

    irep_idt comp_access = component.get_access();
    if(access == clang::AS_public)
    {
      if(comp_access == "private")
        component.set_access("noaccess");
    }
    else if(access == clang::AS_protected)
    {
      if(comp_access == "private")
        component.set_access("noaccess");
      else
        component.set_access("private");
    }
    else if(access == clang::AS_private)
    {
      if(comp_access == "noaccess" || comp_access == "private")
        component.set_access("noaccess");
      else
        component.set_access("private");
    }
    else
      assert(false);
  }
}

void clang_cpp_convertert::get_irep_base_vector(
  const clang::CXXRecordDecl &cxxrd,
  struct_typet &derived_class_type)
{
  irept::subt &bases = derived_class_type.add("bases").get_sub();
  for(const clang::CXXBaseSpecifier &decl : cxxrd.bases())
  {
    typet t;
    if(get_type(decl.getType(), t))
    {
      log_error("Failed to get base type in {}", __func__);
      abort();
    }

    // base type
    assert(t.is_struct() && t.get_bool("#class"));
    struct_union_typet class_t = to_struct_union_type(t);
    t = symbol_typet(tag_prefix + class_t.tag().as_string());
    exprt base("base", t);

    // base access
    base.set("access", get_access(decl.getAccessSpecifier()));

    // base location
    locationt loc;
    const clang::CXXRecordDecl &base_cxxrd =
      *(decl.getType().getTypePtr()->getAsCXXRecordDecl());
    get_location_from_decl(base_cxxrd, loc);
    base.location() = loc;

    bases.push_back(base);
  }
}

void clang_cpp_convertert::get_method_virtual_bases(
  std::set<irep_idt> &virtual_bases,
  const std::string &class_symbol_id,
  const std::string &virtual_name)
{
  assert(current_class_type);
  assert(current_class_type->name().as_string() == class_symbol_id);

  struct_typet::componentst &components = current_class_type->components();
  for(struct_typet::componentst::const_iterator it = components.begin();
      it != components.end();
      it++)
  {
    if(it->get_bool("is_virtual"))
    {
      if(it->get("virtual_name") == virtual_name)
      {
        const code_typet &code_type = to_code_type(it->type());
        assert(code_type.arguments().size() > 0);
        const typet &pointer_type = code_type.arguments()[0].type();
        assert(pointer_type.id() == "pointer");
        virtual_bases.insert(pointer_type.subtype().identifier());
      }
    }
  }
}
