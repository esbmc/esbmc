#include <clang/AST/Attr.h>
#include <clang/AST/QualTypeNames.h>
#include <clang/Index/USRGeneration.h>
#include <clang-c-frontend/clang_c_convert.h>
#include <clang-c-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>

clang_c_convertert::clang_c_convertert(
  contextt &_context,
  std::vector<std::unique_ptr<clang::ASTUnit>> &_ASTs)
  : ASTContext(nullptr),
    context(_context),
    ns(context),
    ASTs(_ASTs),
    current_scope_var_num(1),
    sm(nullptr),
    current_functionDecl(nullptr)
{
}

bool clang_c_convertert::convert()
{
  if(convert_top_level_decl())
    return true;

  return false;
}

bool clang_c_convertert::convert_builtin_types()
{
  // Convert va_list_tag
  const clang::Decl *q_va_list_decl = ASTContext->getVaListTagDecl();
  if(q_va_list_decl)
  {
    exprt dummy;
    if(get_decl(*q_va_list_decl, dummy))
      return true;
  }

  // TODO: clang offers several informations from the target architecture,
  // such as primitive type's size, much like our configt class. We could
  // offer an option in the future to query them from the target system.
  // See clang/Basic/TargetInfo.h for what clang has to offer

  return false;
}

bool clang_c_convertert::convert_top_level_decl()
{
  // Iterate through each translation unit and their global symbols, creating
  // symbols as we go.
  for(auto const &translation_unit : ASTs)
  {
    // Update ASTContext as it changes for each source file
    ASTContext = &(*translation_unit).getASTContext();

    // This is the whole translation unit. We don't represent it internally
    exprt dummy_decl;
    if(get_decl(*ASTContext->getTranslationUnitDecl(), dummy_decl))
      return true;
  }

  assert(current_functionDecl == nullptr);

  return false;
}

// This method convert declarations. They are called when those declarations
// are to be added to the context. If a variable or function is being called
// but then get_decl_expr is called instead
bool clang_c_convertert::get_decl(const clang::Decl &decl, exprt &new_expr)
{
  new_expr = code_skipt();

  switch(decl.getKind())
  {
  // Label declaration
  case clang::Decl::Label:
  {
    const clang::LabelDecl &ld = static_cast<const clang::LabelDecl &>(decl);

    exprt label("label", empty_typet());
    label.identifier(ld.getName().str());
    label.cmt_base_name(ld.getName().str());

    new_expr = label;
    break;
  }

  // Declaration of variables
  case clang::Decl::Var:
  {
    const clang::VarDecl &vd = static_cast<const clang::VarDecl &>(decl);
    return get_var(vd, new_expr);
  }

  // Declaration of function's parameter
  case clang::Decl::ParmVar:
  {
    const clang::ParmVarDecl &param =
      static_cast<const clang::ParmVarDecl &>(decl);
    return get_function_params(param, new_expr);
  }

  // Declaration of functions
  case clang::Decl::Function:
  {
    const clang::FunctionDecl &fd =
      static_cast<const clang::FunctionDecl &>(decl);

    // We can safely ignore any expr generated when parsing C code
    // In C++ mode, methods are initially parsed as function and
    // need to be added to the class scope
    exprt dummy;
    return get_function(fd, dummy);
  }

  // Field inside a struct/union
  case clang::Decl::Field:
  {
    const clang::FieldDecl &fd = static_cast<const clang::FieldDecl &>(decl);

    typet t;
    if(get_type(fd.getType(), t))
      return true;

    std::string id, name;
    get_decl_name(fd, name, id);

    struct_union_typet::componentt comp(id, name, t);
    if(fd.isBitField())
    {
      exprt width;
      if(get_expr(*fd.getBitWidth(), width))
        return true;

      comp.type().width(width.cformat());
      comp.type().set("#bitfield", "true");
    }

    new_expr.swap(comp);
    break;
  }

  case clang::Decl::IndirectField:
  {
    const clang::IndirectFieldDecl &fd =
      static_cast<const clang::IndirectFieldDecl &>(decl);

    typet t;
    if(get_type(fd.getType(), t))
      return true;

    std::string id, name;
    get_decl_name(*fd.getAnonField(), name, id);

    struct_union_typet::componentt comp(id, name, t);
    if(fd.getAnonField()->isBitField())
    {
      exprt width;
      if(get_expr(*fd.getAnonField()->getBitWidth(), width))
        return true;

      comp.type().set("#bitfield", "true");
      comp.type().width(width.cformat());
    }

    new_expr.swap(comp);
    break;
  }

  // A record is a struct/union/class/enum
  case clang::Decl::Record:
  {
    const clang::RecordDecl &record =
      static_cast<const clang::RecordDecl &>(decl);

    if(get_struct_union_class(record))
      return true;

    break;
  }

  case clang::Decl::TranslationUnit:
  {
    const clang::TranslationUnitDecl &tu =
      static_cast<const clang::TranslationUnitDecl &>(decl);

    for(auto const &decl : tu.decls())
    {
      // This is a global declaration (variable, function, struct, etc)
      // We don't need the exprt, it will be automatically added to the
      // context
      exprt dummy_decl;
      if(get_decl(*decl, dummy_decl))
        return true;
    }

    break;
  }

  // This is an empty declaration. An lost semicolon on the
  // code is an empty declaration
  case clang::Decl::Empty:

  // If this fails, clang will not generate the ASTs, we can
  // safely skip it
  case clang::Decl::StaticAssert:

  // Enum declaration and values, we can safely skip them as
  // any occurrence of those will be converted to int type (enum)
  // or integer value (enum constant)
  case clang::Decl::Enum:
  case clang::Decl::EnumConstant:

  // Typedef declaration, we can ignore this; clang will give us
  // the underlying type defined by the typedef, so we don't need
  // to add them to the context
  case clang::Decl::Typedef:
    break;

  default:
    std::cerr << "**** ERROR: ";
    std::cerr << "Unrecognized / unimplemented clang declaration "
              << decl.getDeclKindName() << std::endl;
    decl.dumpColor();
    return true;
  }

  return false;
}

bool clang_c_convertert::get_struct_union_class(const clang::RecordDecl &rd)
{
  if(rd.isClass())
  {
    std::cerr << "Class is not supported yet" << std::endl;
    return true;
  }
  else if(rd.isInterface())
  {
    std::cerr << "Interface is not supported yet" << std::endl;
    return true;
  }

  struct_union_typet t;
  if(rd.isStruct())
    t = struct_typet();
  else if(rd.isUnion())
    t = union_typet();
  else
    // This should never be reached
    abort();

  std::string id, name;
  get_decl_name(rd, name, id);
  t.tag(name);

  locationt location_begin;
  get_location_from_decl(rd, location_begin);

  symbolt symbol;
  get_default_symbol(
    symbol,
    get_modulename_from_path(location_begin.file().as_string()),
    t,
    name,
    id,
    location_begin);

  // Save the struct/union/class type address and name to the type map
  std::string symbol_name = symbol.id.as_string();

  std::size_t address = reinterpret_cast<std::size_t>(rd.getFirstDecl());
  type_map[address] = symbol_name;

  symbol.is_type = true;

  // We have to add the struct/union/class to the context before converting its
  // fields because there might be recursive struct/union/class (pointers) and
  // the code at get_type, case clang::Type::Record, needs to find the correct
  // type (itself). Note that the type is incomplete at this stage, it doesn't
  // contain the fields, which are added to the symbol later on this method.
  move_symbol_to_context(symbol);

  // Don't continue to parse if it doesn't have a complete definition
  // Try to get the definition
  clang::RecordDecl *rd_def = rd.getDefinition();
  if(!rd_def)
    return false;

  // Now get the symbol back to continue the conversion
  symbolt &added_symbol = *context.find_symbol(symbol_name);

  if(get_struct_union_class_fields(*rd_def, t))
    return true;

  // Check for packed specifier.
  if(rd_def->hasAttrs())
  {
    const auto &attrs = rd_def->getAttrs();
    for(const auto &attr : attrs)
      if(attr->getKind() == clang::attr::Packed)
        t.set("packed", "true");
  }

  added_symbol.type = has_bitfields(t) ? fix_bitfields(t) : t;

  return false;
}

bool clang_c_convertert::get_struct_union_class_fields(
  const clang::RecordDecl &recordd,
  struct_union_typet &type)
{
  // First, parse the fields
  for(auto const &field : recordd.fields())
  {
    struct_typet::componentt comp;
    if(get_decl(*field, comp))
      return true;

    // Don't add fields that have global storage (e.g., static)
    if(const clang::VarDecl *nd = llvm::dyn_cast<clang::VarDecl>(field))
      if(nd->hasGlobalStorage())
        continue;

    type.components().push_back(comp);
  }

  return false;
}

bool clang_c_convertert::get_struct_union_class_methods(
  const clang::RecordDecl &recordd,
  struct_union_typet &type)
{
  // We don't add methods to the struct in C
  return false;
}

bool clang_c_convertert::get_var(const clang::VarDecl &vd, exprt &new_expr)
{
  // Get type
  typet t;
  if(get_type(vd.getType(), t))
    return true;

  // Check if we annotated it to be have an infinity size
  if(vd.hasAttrs())
  {
    for(auto const &attr : vd.getAttrs())
    {
      if(const auto *a = llvm::dyn_cast<clang::AnnotateAttr>(attr))
      {
        if(a->getAnnotation().str() == "__ESBMC_inf_size")
        {
          assert(t.is_array());
          t.size(exprt("infinity", uint_type()));
        }
      }
    }
  }

  std::string id, name;
  get_decl_name(vd, name, id);

  locationt location_begin;
  get_location_from_decl(vd, location_begin);

  symbolt symbol;
  get_default_symbol(
    symbol,
    get_modulename_from_path(location_begin.file().as_string()),
    t,
    name,
    id,
    location_begin);

  symbol.lvalue = true;
  symbol.static_lifetime =
    (vd.getStorageClass() == clang::SC_Static) || vd.hasGlobalStorage();
  symbol.is_extern = vd.hasExternalStorage();
  symbol.file_local = (vd.getStorageClass() == clang::SC_Static) ||
                      (!vd.isExternallyVisible() && !vd.hasGlobalStorage());

  if(symbol.static_lifetime && !symbol.is_extern && !vd.hasInit())
  {
    // Initialize with zero value, if the symbol has initial value,
    // it will be add later on this method
    symbol.value = gen_zero(t, true);
    symbol.value.zero_initializer(true);
  }

  // We have to add the symbol before converting the initial assignment
  // because we might have something like 'int x = x + 1;' which is
  // completely wrong but allowed by the language
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  code_declt decl(symbol_expr(added_symbol));

  if(vd.hasInit())
  {
    exprt val;
    if(get_expr(*vd.getInit(), val))
      return true;

    gen_typecast(ns, val, t);

    added_symbol.value = val;
    decl.operands().push_back(val);
  }

  decl.location() = location_begin;

  new_expr = decl;
  return false;
}

bool clang_c_convertert::get_function(
  const clang::FunctionDecl &fd,
  exprt &new_expr)
{
  // Don't convert if clang thinks that the functions was implicitly converted
  if(fd.isImplicit())
    return false;

  // If the function is not defined but this is not the definition, skip it
  if(fd.isDefined() && !fd.isThisDeclarationADefinition())
    return false;

  // Save old_functionDecl, to be restored at the end of this method
  const clang::FunctionDecl *old_functionDecl = current_functionDecl;
  current_functionDecl = &fd;

  // TODO: use fd.isMain to flag and check the flag on clang_c_adjust_expr
  // to saner way to add argc/argv/envp

  // Set initial variable name, it will be used for variables' name
  // This will be reset every time a function is parsed
  current_scope_var_num = 1;

  // Build function's type
  code_typet type;

  // Return type
  if(get_type(fd.getReturnType(), type.return_type()))
    return true;

  if(fd.isVariadic())
    type.make_ellipsis();

  if(fd.isInlined())
    type.inlined(true);

  locationt location_begin;
  get_location_from_decl(fd, location_begin);

  std::string id, name;
  get_decl_name(fd, name, id);

  symbolt symbol;
  get_default_symbol(
    symbol,
    get_modulename_from_path(location_begin.file().as_string()),
    type,
    name,
    id,
    location_begin);

  std::string symbol_name = symbol.id.as_string();

  symbol.lvalue = true;
  symbol.is_extern = fd.getStorageClass() == clang::SC_Extern ||
                     fd.getStorageClass() == clang::SC_PrivateExtern;
  symbol.file_local = (fd.getStorageClass() == clang::SC_Static);

  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // We convert the parameters first so their symbol are added to context
  // before converting the body, as they may appear on the function body
  for(auto const &pdecl : fd.parameters())
  {
    code_typet::argumentt param;
    if(get_function_params(*pdecl, param))
      return true;

    type.arguments().push_back(param);
  }

  // Apparently, if the type has no arguments, we assume ellipsis
  if(!type.arguments().size())
    type.make_ellipsis();

  added_symbol.type = type;

  // We need: a type, a name, and an optional body
  if(fd.hasBody())
  {
    exprt body_exprt;
    if(get_expr(*fd.getBody(), body_exprt))
      return true;

    added_symbol.value = body_exprt;
  }

  // Restore old functionDecl
  current_functionDecl = old_functionDecl;

  return false;
}

bool clang_c_convertert::get_function_params(
  const clang::ParmVarDecl &pd,
  exprt &param)
{
  typet param_type;
  if(get_type(pd.getOriginalType(), param_type))
    return true;

  if(param_type.is_array())
  {
    param_type.id("pointer");
    param_type.remove("size");
    param_type.remove("#constant");
  }

  std::string id, name;
  get_decl_name(pd, name, id);

  param = code_typet::argumentt();
  param.type() = param_type;
  param.cmt_base_name(name);

  // If the name is empty, this is an function definition that we don't
  // need to worry about as the function params name's will be defined
  // when the function is defined, the exprt is filled for the sake of
  // beautification
  if(name.empty())
    return false;

  locationt location_begin;
  get_location_from_decl(pd, location_begin);

  param.cmt_identifier(id);
  param.location() = location_begin;

  // TODO: we can remove the following code once irep1 is dead, there
  // is no need to add the function argument to the symbol table,
  // as nothing relies on it. However, if we remove this now, the migrate
  // code will wrongly assume the symbol to be level1, as it generates
  // level0 symbol only if they are already on the context
  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    get_modulename_from_path(location_begin.file().as_string()),
    param_type,
    name,
    id,
    location_begin);

  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  const clang::FunctionDecl &fd =
    static_cast<const clang::FunctionDecl &>(*pd.getParentFunctionOrMethod());

  // If the function is not defined, we don't need to add it's parameter
  // to the context, they will never be used
  if(!fd.isDefined())
    return false;

  move_symbol_to_context(param_symbol);
  return false;
}

bool clang_c_convertert::get_type(
  const clang::QualType &q_type,
  typet &new_type)
{
  const clang::Type &the_type = *q_type.getTypePtrOrNull();
  if(get_type(the_type, new_type))
    return true;

  if(q_type.isConstQualified())
    new_type.cmt_constant(true);

  if(q_type.isVolatileQualified())
    new_type.cmt_volatile(true);

  if(q_type.isRestrictQualified())
    new_type.restricted(true);

  return false;
}

bool clang_c_convertert::get_type(const clang::Type &the_type, typet &new_type)
{
  switch(the_type.getTypeClass())
  {
  // Builtin types like integer
  case clang::Type::Builtin:
  {
    const clang::BuiltinType &bt =
      static_cast<const clang::BuiltinType &>(the_type);

    if(get_builtin_type(bt, new_type))
      return true;

    break;
  }

  // Types using parenthesis, e.g. int (a);
  case clang::Type::Paren:
  {
    const clang::ParenType &pt =
      static_cast<const clang::ParenType &>(the_type);

    if(get_type(pt.getInnerType(), new_type))
      return true;

    break;
  }

  // Pointer types
  case clang::Type::Pointer:
  {
    const clang::PointerType &pt =
      static_cast<const clang::PointerType &>(the_type);
    const clang::QualType &pointee = pt.getPointeeType();

    typet sub_type;
    if(get_type(pointee, sub_type))
      return true;

    // Special case, pointers to structs/unions/classes must not
    // have a copy of it, but a reference to the type
    // TODO: classes
    if(sub_type.is_struct() || sub_type.is_union())
    {
      struct_union_typet t = to_struct_union_type(sub_type);
      sub_type = symbol_typet("tag-" + t.tag().as_string());
    }

    new_type = gen_pointer_type(sub_type);
    break;
  }

  // Types adjusted by the semantic engine
  case clang::Type::Decayed:
  {
    const clang::DecayedType &pt =
      static_cast<const clang::DecayedType &>(the_type);

    if(get_type(pt.getDecayedType(), new_type))
      return true;

    break;
  }

  // Array with constant size, e.g., int a[3];
  case clang::Type::ConstantArray:
  {
    const clang::ConstantArrayType &arr =
      static_cast<const clang::ConstantArrayType &>(the_type);

    llvm::APInt val = arr.getSize();
    if(val.getBitWidth() > 64)
    {
      std::cerr << "ESBMC currently does not support integers bigger "
                   "than 64 bits"
                << std::endl;
      return true;
    }

    typet the_type;
    if(get_type(arr.getElementType(), the_type))
      return true;

    new_type = array_typet(
      the_type,
      constant_exprt(
        integer2binary(val.getSExtValue(), bv_width(int_type())),
        integer2string(val.getSExtValue()),
        int_type()));
    break;
  }

  // Array with undefined type, as in function args
  case clang::Type::IncompleteArray:
  {
    const clang::IncompleteArrayType &arr =
      static_cast<const clang::IncompleteArrayType &>(the_type);

    typet sub_type;
    if(get_type(arr.getElementType(), sub_type))
      return true;

    new_type = array_typet(sub_type, gen_one(index_type()));
    break;
  }

  // Array with variable size, e.g., int a[n];
  case clang::Type::VariableArray:
  {
    const clang::VariableArrayType &arr =
      static_cast<const clang::VariableArrayType &>(the_type);

    // If the size expression is null, we assume empty
    if(auto const *s = arr.getSizeExpr())
    {
      exprt size_expr;
      if(get_expr(*s, size_expr))
        return true;

      typet subtype;
      if(get_type(arr.getElementType(), subtype))
        return true;

      new_type = array_typet(subtype, size_expr);
    }
    else
      new_type = empty_typet();
    break;
  }

  // Those two here appears when we make a function call, e.g:
  // FunctionNoProto: int x = fun()
  // FunctionProto: int x = fun(a, b)
  case clang::Type::FunctionProto:
  {
    const clang::FunctionProtoType &func =
      static_cast<const clang::FunctionProtoType &>(the_type);

    code_typet type;

    // Return type
    const clang::QualType ret_type = func.getReturnType();

    typet return_type;
    if(get_type(ret_type, return_type))
      return true;

    type.return_type() = return_type;

    for(auto const &ptype : func.getParamTypes())
    {
      typet param_type;
      if(get_type(ptype, param_type))
        return true;

      type.arguments().emplace_back(param_type);
    }

    // Apparently, if the type has no arguments, we assume ellipsis
    if(!type.arguments().size() || func.isVariadic())
      type.make_ellipsis();

    new_type = type;
    break;
  }

  case clang::Type::FunctionNoProto:
  {
    const clang::FunctionNoProtoType &func =
      static_cast<const clang::FunctionNoProtoType &>(the_type);

    code_typet type;

    // Return type
    const clang::QualType ret_type = func.getReturnType();

    typet return_type;
    if(get_type(ret_type, return_type))
      return true;

    type.return_type() = return_type;

    // Apparently, if the type has no arguments, we assume ellipsis
    if(!type.arguments().size())
      type.make_ellipsis();

    new_type = type;
    break;
  }

  // Typedef type definition
  case clang::Type::Typedef:
  {
    const clang::TypedefType &pt =
      static_cast<const clang::TypedefType &>(the_type);

    clang::QualType q_typedef_type =
      pt.getDecl()->getUnderlyingType().getCanonicalType();

    if(get_type(q_typedef_type, new_type))
      return true;

    break;
  }

  case clang::Type::Record:
  {
    const clang::RecordDecl &rd =
      *(static_cast<const clang::RecordType &>(the_type)).getDecl();

    if(rd.isClass())
    {
      std::cerr << "Class Type is not supported yet" << std::endl;
      return true;
    }

    // Search for the type on the type map
    type_mapt::iterator it;
    if(search_add_type_map(rd, it))
      return true;

    symbolt &s = *context.find_symbol(it->second);
    new_type = s.type;

    break;
  }

  case clang::Type::Enum:
  {
    new_type = enum_type();
    break;
  }

  case clang::Type::Elaborated:
  {
    const clang::ElaboratedType &et =
      static_cast<const clang::ElaboratedType &>(the_type);

    if(get_type(et.getNamedType(), new_type))
      return true;
    break;
  }

  case clang::Type::TypeOfExpr:
  {
    const clang::TypeOfExprType &tofe =
      static_cast<const clang::TypeOfExprType &>(the_type);

    if(get_type(tofe.desugar(), new_type))
      return true;

    break;
  }

  case clang::Type::TypeOf:
  {
    const clang::TypeOfType &toft =
      static_cast<const clang::TypeOfType &>(the_type);

    if(get_type(toft.desugar(), new_type))
      return true;

    break;
  }

  case clang::Type::LValueReference:
  {
    const clang::LValueReferenceType &lvrt =
      static_cast<const clang::LValueReferenceType &>(the_type);

    if(get_type(lvrt.getPointeeTypeAsWritten(), new_type))
      return true;

    break;
  }

  case clang::Type::Attributed:
  {
    const clang::AttributedType &att =
      static_cast<const clang::AttributedType &>(the_type);

    if(get_type(att.desugar(), new_type))
      return true;

    break;
  }

  case clang::Type::Decltype:
  {
    const clang::DecltypeType &dt =
      static_cast<const clang::DecltypeType &>(the_type);

    if(get_type(dt.getUnderlyingType(), new_type))
      return true;

    break;
  }

  case clang::Type::Atomic:
  {
    const clang::AtomicType &dt =
      static_cast<const clang::AtomicType &>(the_type);

    // FIXME: we need some representation of atomic types in irep2
    if(get_type(dt.getValueType(), new_type))
      return true;

    break;
  }

  case clang::Type::Auto:
  {
    const clang::AutoType &at = static_cast<const clang::AutoType &>(the_type);

    if(get_type(at.desugar(), new_type))
      return true;

    break;
  }

  default:
    std::cerr << "Conversion of unsupported clang type: \"";
    std::cerr << the_type.getTypeClassName() << std::endl;
    the_type.dump();
    return true;
  }

  return false;
}

bool clang_c_convertert::get_builtin_type(
  const clang::BuiltinType &bt,
  typet &new_type)
{
  std::string c_type;

  switch(bt.getKind())
  {
  case clang::BuiltinType::Void:
    new_type = empty_typet();
    c_type = "void";
    break;

  case clang::BuiltinType::Bool:
    new_type = bool_type();
    c_type = "bool";
    break;

  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::UChar:
    new_type = unsigned_char_type();
    c_type = "unsigned_char";
    break;

  case clang::BuiltinType::WChar_U:
    new_type = unsigned_wchar_type();
    c_type = "unsigned_wchar_t";
    break;

  case clang::BuiltinType::Char16:
    new_type = char16_type();
    c_type = "char16_t";
    break;

  case clang::BuiltinType::Char32:
    new_type = char32_type();
    c_type = "char32_t";
    break;

  case clang::BuiltinType::UShort:
    new_type = unsigned_short_int_type();
    c_type = "unsigned_short";
    break;

  case clang::BuiltinType::UInt:
    new_type = uint_type();
    c_type = "unsigned_int";
    break;

  case clang::BuiltinType::ULong:
    new_type = long_uint_type();
    c_type = "unsigned_long";
    break;

  case clang::BuiltinType::ULongLong:
    new_type = long_long_uint_type();
    c_type = "unsigned_long_long";
    break;

  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::SChar:
    new_type = signed_char_type();
    c_type = "signed_char";
    break;

  case clang::BuiltinType::WChar_S:
    new_type = wchar_type();
    c_type = "wchar_t";
    break;

  case clang::BuiltinType::Short:
    new_type = signed_short_int_type();
    c_type = "signed_short";
    break;

  case clang::BuiltinType::Int:
    new_type = int_type();
    c_type = "signed_int";
    break;

  case clang::BuiltinType::Long:
    new_type = long_int_type();
    c_type = "signed_long";
    break;

  case clang::BuiltinType::LongLong:
    new_type = long_long_int_type();
    c_type = "signed_long_long";
    break;

  case clang::BuiltinType::Half:
    new_type = half_float_type();
    c_type = "_Float16";
    break;

  case clang::BuiltinType::Float:
    new_type = float_type();
    c_type = "float";
    break;

  case clang::BuiltinType::Double:
    new_type = double_type();
    c_type = "double";
    break;

  case clang::BuiltinType::Float128:
  case clang::BuiltinType::LongDouble:
    new_type = long_double_type();
    c_type = "long_double";
    break;

  case clang::BuiltinType::Int128:
  case clang::BuiltinType::UInt128:
    // Various simplification / big-int related things use uint64_t's...
    std::cerr << "ESBMC currently does not support integers bigger "
              << "than 64 bits" << std::endl;
    bt.dump();
    return true;

  default:
    std::cerr << "Unrecognized clang builtin type "
              << bt.getName(clang::PrintingPolicy(clang::LangOptions())).str()
              << std::endl;
    bt.dump();
    return true;
  }

  new_type.set("#cpp_type", c_type);
  return false;
}

bool clang_c_convertert::get_expr(const clang::Stmt &stmt, exprt &new_expr)
{
  locationt location;
  get_start_location_from_stmt(stmt, location);

  switch(stmt.getStmtClass())
  {
  /*
       The following enum values are the the expr of a program,
       defined on the Expr class
    */

  // Objects that are implicit defined on the code syntax.
  // One example is the gcc ternary operator, which can be:
  // _Bool a = 1 ? : 0; is equivalent to _Bool a = 1 ? 1 : 0;
  // The 'then' expr is an opaque value equal to the ternary's
  // condition
  case clang::Stmt::OpaqueValueExprClass:
  {
    const clang::OpaqueValueExpr &opaque_expr =
      static_cast<const clang::OpaqueValueExpr &>(stmt);

    if(get_expr(*opaque_expr.getSourceExpr(), new_expr))
      return true;

    break;
  }

  // Reference to a declared object, such as functions or variables
  case clang::Stmt::DeclRefExprClass:
  {
    const clang::DeclRefExpr &decl =
      static_cast<const clang::DeclRefExpr &>(stmt);

    const clang::Decl &dcl = static_cast<const clang::Decl &>(*decl.getDecl());

    if(get_decl_ref(dcl, new_expr))
      return true;

    break;
  }

  // Predefined MACROS as __func__ or __PRETTY_FUNCTION__
  case clang::Stmt::PredefinedExprClass:
  {
    const clang::PredefinedExpr &pred_expr =
      static_cast<const clang::PredefinedExpr &>(stmt);

    if(convert_string_literal(*pred_expr.getFunctionName(), new_expr))
      return true;

    break;
  }

  // An integer value
  case clang::Stmt::IntegerLiteralClass:
  {
    const clang::IntegerLiteral &integer_literal =
      static_cast<const clang::IntegerLiteral &>(stmt);

    if(convert_integer_literal(integer_literal, new_expr))
      return true;

    break;
  }

  // A character such 'a'
  case clang::Stmt::CharacterLiteralClass:
  {
    const clang::CharacterLiteral &char_literal =
      static_cast<const clang::CharacterLiteral &>(stmt);

    if(convert_character_literal(char_literal, new_expr))
      return true;

    break;
  }

  // A float value
  case clang::Stmt::FloatingLiteralClass:
  {
    const clang::FloatingLiteral &floating_literal =
      static_cast<const clang::FloatingLiteral &>(stmt);

    if(convert_float_literal(floating_literal, new_expr))
      return true;

    break;
  }

  // A string
  case clang::Stmt::StringLiteralClass:
  {
    const clang::StringLiteral &string_literal =
      static_cast<const clang::StringLiteral &>(stmt);

    if(convert_string_literal(string_literal, new_expr))
      return true;

    break;
  }

  // This is an expr surrounded by parenthesis, we'll ignore it for
  // now, and check its subexpression
  case clang::Stmt::ParenExprClass:
  {
    const clang::ParenExpr &p = static_cast<const clang::ParenExpr &>(stmt);

    if(get_expr(*p.getSubExpr(), new_expr))
      return true;

    break;
  }

  // An unary operator such as +a, -a, *a or &a
  case clang::Stmt::UnaryOperatorClass:
  {
    const clang::UnaryOperator &uniop =
      static_cast<const clang::UnaryOperator &>(stmt);

    if(get_unary_operator_expr(uniop, new_expr))
      return true;

    break;
  }

  // An array subscript operation, such as a[1]
  case clang::Stmt::ArraySubscriptExprClass:
  {
    const clang::ArraySubscriptExpr &arr =
      static_cast<const clang::ArraySubscriptExpr &>(stmt);

    typet t;
    if(get_type(arr.getType(), t))
      return true;

    exprt array;
    if(get_expr(*arr.getBase(), array))
      return true;

    exprt pos;
    if(get_expr(*arr.getIdx(), pos))
      return true;

    new_expr = index_exprt(array, pos, t);
    break;
  }

  // Support for __builtin_offsetof();
  case clang::Stmt::OffsetOfExprClass:
  {
    const clang::OffsetOfExpr &offset =
      static_cast<const clang::OffsetOfExpr &>(stmt);

    // Use clang to calculate offsetof
    clang::Expr::EvalResult result;
    bool res = offset.EvaluateAsInt(result, *ASTContext);
    if(!res)
    {
      std::cerr << "Clang could not calculate offset" << std::endl;
      offset.dumpColor();
      return true;
    }

    new_expr = constant_exprt(
      integer2binary(result.Val.getInt().getSExtValue(), bv_width(uint_type())),
      integer2string(result.Val.getInt().getSExtValue()),
      uint_type());
    break;
  }

  case clang::Stmt::UnaryExprOrTypeTraitExprClass:
  {
    const clang::UnaryExprOrTypeTraitExpr &unary =
      static_cast<const clang::UnaryExprOrTypeTraitExpr &>(stmt);

    // Use clang to calculate alignof
    if(unary.getKind() == clang::UETT_AlignOf)
    {
      clang::Expr::EvalResult result;
      if(!unary.EvaluateAsInt(result, *ASTContext))
        return true;

      new_expr = constant_exprt(
        integer2binary(
          result.Val.getInt().getZExtValue(), bv_width(uint_type())),
        integer2string(result.Val.getInt().getZExtValue()),
        uint_type());
    }
    else
    {
      assert(unary.getKind() == clang::UETT_SizeOf);

      typet t;
      if(get_type(unary.getType(), t))
        return true;

      new_expr = exprt("sizeof", t);
    }

    typet size_type;
    if(get_type(unary.getTypeOfArgument(), size_type))
      return true;

    if(size_type.is_struct() || size_type.is_union())
    {
      struct_union_typet t = to_struct_union_type(size_type);
      size_type = symbol_typet("tag-" + t.tag().as_string());
    }

    new_expr.set("#c_sizeof_type", size_type);
    break;
  }

  // A function call expr
  // It can be undefined here, the symbol will be added in
  // adjust_expr::adjust_side_effect_function_call
  case clang::Stmt::CallExprClass:
  {
    const clang::CallExpr &function_call =
      static_cast<const clang::CallExpr &>(stmt);

    const clang::Stmt *callee = function_call.getCallee();

    exprt callee_expr;
    if(get_expr(*callee, callee_expr))
      return true;

    typet type;
    if(get_type(function_call.getType(), type))
      return true;

    side_effect_expr_function_callt call;
    call.function() = callee_expr;
    call.type() = type;

    for(const clang::Expr *arg : function_call.arguments())
    {
      exprt single_arg;
      if(get_expr(*arg, single_arg))
        return true;

      call.arguments().push_back(single_arg);
    }

    new_expr = call;
    break;
  }

  case clang::Stmt::MemberExprClass:
  {
    const clang::MemberExpr &member =
      static_cast<const clang::MemberExpr &>(stmt);

    exprt base;
    if(get_expr(*member.getBase(), base))
      return true;

    exprt comp;
    if(get_decl(*member.getMemberDecl(), comp))
      return true;

    new_expr = member_exprt(base, comp.name(), comp.type());

    // Potentially fix up a bitfield
    typet base_type = base.type();
    if(base.type().id() == "pointer")
      base_type = base.type().subtype();

    typet fixed_version;
    if(has_bitfields(base_type, &fixed_version))
    {
      // Look up record for this struct kind. Should be converted.
      const auto bmit = bitfield_mappings.find(fixed_version);
      assert(bmit != bitfield_mappings.end());
      const auto &backmap = bmit->second;

      auto it = backmap.find(comp.name());
      if(it != backmap.end())
        rewrite_bitfield_member(new_expr, it->second);
    }
    break;
  }

  case clang::Stmt::CompoundLiteralExprClass:
  {
    const clang::CompoundLiteralExpr &compound =
      static_cast<const clang::CompoundLiteralExpr &>(stmt);

    exprt initializer;
    if(get_expr(*compound.getInitializer(), initializer))
      return true;

    new_expr = initializer;
    break;
  }

  case clang::Stmt::AddrLabelExprClass:
  {
    const clang::AddrLabelExpr &addrlabelExpr =
      static_cast<const clang::AddrLabelExpr &>(stmt);

    exprt label;
    if(get_decl(*addrlabelExpr.getLabel(), label))
      return true;

    new_expr = address_of_exprt(label);
    break;
  }

  case clang::Stmt::StmtExprClass:
  {
    const clang::StmtExpr &stmtExpr =
      static_cast<const clang::StmtExpr &>(stmt);

    typet t;
    if(get_type(stmtExpr.getType(), t))
      return true;

    exprt subStmt;
    if(get_expr(*stmtExpr.getSubStmt(), subStmt))
      return true;

    side_effect_exprt stmt_expr("statement_expression", t);
    stmt_expr.copy_to_operands(subStmt);

    new_expr = stmt_expr;
    break;
  }

  case clang::Stmt::GNUNullExprClass:
  {
    const clang::GNUNullExpr &gnun =
      static_cast<const clang::GNUNullExpr &>(stmt);

    typet t;
    if(get_type(gnun.getType(), t))
      return true;

    new_expr = gen_zero(t);
    break;
  }
  // Casts expression:
  // Implicit: float f = 1; equivalent to float f = (float) 1;
  // CStyle: int a = (int) 3.0;
  case clang::Stmt::ImplicitCastExprClass:
  case clang::Stmt::CStyleCastExprClass:
  {
    const clang::CastExpr &cast = static_cast<const clang::CastExpr &>(stmt);

    if(get_cast_expr(cast, new_expr))
      return true;

    break;
  }

  // Binary expression such as a+1, a-1 and assignments
  case clang::Stmt::BinaryOperatorClass:
  case clang::Stmt::CompoundAssignOperatorClass:
  {
    const clang::BinaryOperator &binop =
      static_cast<const clang::BinaryOperator &>(stmt);

    if(get_binary_operator_expr(binop, new_expr))
      return true;

    break;
  }

  // This is the ternary if
  case clang::Stmt::ConditionalOperatorClass:
  {
    const clang::ConditionalOperator &ternary_if =
      static_cast<const clang::ConditionalOperator &>(stmt);

    exprt cond;
    if(get_expr(*ternary_if.getCond(), cond))
      return true;

    exprt then;
    if(get_expr(*ternary_if.getTrueExpr(), then))
      return true;

    exprt else_expr;
    if(get_expr(*ternary_if.getFalseExpr(), else_expr))
      return true;

    typet t;
    if(get_type(ternary_if.getType(), t))
      return true;

    exprt if_expr("if", t);
    if_expr.copy_to_operands(cond, then, else_expr);

    new_expr = if_expr;
    break;
  }

  // This is the gcc's ternary if extension
  case clang::Stmt::BinaryConditionalOperatorClass:
  {
    const clang::BinaryConditionalOperator &ternary_if =
      static_cast<const clang::BinaryConditionalOperator &>(stmt);

    exprt cond;
    if(get_expr(*ternary_if.getCond(), cond))
      return true;

    exprt else_expr;
    if(get_expr(*ternary_if.getFalseExpr(), else_expr))
      return true;

    typet t;
    if(get_type(ternary_if.getType(), t))
      return true;

    side_effect_exprt gcc_ternary("gcc_conditional_expression", t);
    gcc_ternary.copy_to_operands(cond, else_expr);

    new_expr = gcc_ternary;
    break;
  }

  // An initialize statement, such as int a[3] = {1, 2, 3}
  case clang::Stmt::InitListExprClass:
  {
    const clang::InitListExpr &init_stmt =
      static_cast<const clang::InitListExpr &>(stmt);

    typet t;
    if(get_type(init_stmt.getType(), t))
      return true;

    // If it _was_ a bitfield, treat is as non-mangled for the moment.
    if(has_bitfields(t))
      t = bitfield_orig_type_map[t];

    exprt inits;

    // Structs/unions/arrays put the initializer on operands
    if(t.is_struct() || t.is_union() || t.is_array())
    {
      inits = gen_zero(t);

      unsigned int num = init_stmt.getNumInits();
      for(unsigned int i = 0; i < num; i++)
      {
        exprt init;
        if(get_expr(*init_stmt.getInit(i), init))
          return true;

        typet elem_type;

        if(t.is_struct() || t.is_union())
          elem_type = to_struct_union_type(t).components()[i].type();
        else
          elem_type = to_array_type(t).subtype();

        gen_typecast(ns, init, elem_type);

        inits.operands().at(i) = init;
      }

      // Handle bitfields again..
      if(has_bitfields(t))
      {
        fix_constant_bitfields(inits);
        t = bitfield_orig_type_map[t];
      }

      // If this expression is initializing an union, we should
      // set which field is being initialized
      if(t.is_union())
      {
        to_union_expr(inits).set_component_name(
          init_stmt.getInitializedFieldInUnion()->getName().str());
      }
    }
    else
    {
      assert(init_stmt.getNumInits() == 1);
      if(get_expr(*init_stmt.getInit(0), inits))
        return true;
    }

    new_expr = inits;
    break;
  }

  case clang::Stmt::ImplicitValueInitExprClass:
  {
    const clang::ImplicitValueInitExpr &init_stmt =
      static_cast<const clang::ImplicitValueInitExpr &>(stmt);

    typet t;
    if(get_type(init_stmt.getType(), t))
      return true;

    new_expr = gen_zero(t);
    break;
  }

  case clang::Stmt::GenericSelectionExprClass:
  {
    const clang::GenericSelectionExpr &gen =
      static_cast<const clang::GenericSelectionExpr &>(stmt);

    if(get_expr(*gen.getResultExpr(), new_expr))
      return true;

    break;
  }

  case clang::Stmt::VAArgExprClass:
  {
    const clang::VAArgExpr &vaa = static_cast<const clang::VAArgExpr &>(stmt);

    exprt expr;
    if(get_expr(*vaa.getSubExpr(), expr))
      return true;

    typet t;
    if(get_type(vaa.getType(), t))
      return true;

    exprt vaa_expr("builtin_va_arg", t);
    vaa_expr.copy_to_operands(expr);

    new_expr = vaa_expr;
    break;
  }

  case clang::Stmt::ConstantExprClass:
  {
    const clang::ConstantExpr &c =
      static_cast<const clang::ConstantExpr &>(stmt);

    if(get_expr(*c.getSubExpr(), new_expr))
      return true;

    break;
  }

  /*
    The following enum values are the basic elements of a program,
    defined on the Stmt class
  */

  // Declaration of variables, it is created as a decl-block to
  // allow declarations like int a,b;
  case clang::Stmt::DeclStmtClass:
  {
    const clang::DeclStmt &decl = static_cast<const clang::DeclStmt &>(stmt);

    const auto &declgroup = decl.getDeclGroup();

    codet decls("decl-block");
    for(auto it : declgroup)
    {
      exprt single_decl;
      if(get_decl(*it, single_decl))
        return true;

      decls.operands().push_back(single_decl);
    }

    new_expr = decls;
    break;
  }

  // A compound statement is a scope/block
  case clang::Stmt::CompoundStmtClass:
  {
    const clang::CompoundStmt &compound_stmt =
      static_cast<const clang::CompoundStmt &>(stmt);

    code_blockt block;
    for(auto const &stmt : compound_stmt.body())
    {
      exprt statement;
      if(get_expr(*stmt, statement))
        return true;

      convert_expression_to_code(statement);
      block.operands().push_back(statement);
    }

    // Set the end location for blocks
    locationt location_end;
    get_final_location_from_stmt(stmt, location_end);

    block.end_location(location_end);

    new_expr = block;
    break;
  }

  // A case statement inside a switch. The detail here is that we
  // construct it as a label
  case clang::Stmt::CaseStmtClass:
  {
    const clang::CaseStmt &case_stmt =
      static_cast<const clang::CaseStmt &>(stmt);

    exprt value;
    if(get_expr(*case_stmt.getLHS(), value))
      return true;

    exprt sub_stmt;
    if(get_expr(*case_stmt.getSubStmt(), sub_stmt))
      return true;

    code_switch_caset switch_case;
    switch_case.case_op() = value;

    convert_expression_to_code(sub_stmt);
    switch_case.code() = to_code(sub_stmt);

    new_expr = switch_case;
    break;
  }

  // A default statement inside a switch. Same as before, we construct
  // as a label, the difference is that we set default to true
  case clang::Stmt::DefaultStmtClass:
  {
    const clang::DefaultStmt &default_stmt =
      static_cast<const clang::DefaultStmt &>(stmt);

    exprt sub_stmt;
    if(get_expr(*default_stmt.getSubStmt(), sub_stmt))
      return true;

    code_switch_caset switch_case;
    switch_case.set_default(true);

    convert_expression_to_code(sub_stmt);
    switch_case.code() = to_code(sub_stmt);

    new_expr = switch_case;
    break;
  }

  // A label on the program
  case clang::Stmt::LabelStmtClass:
  {
    const clang::LabelStmt &label_stmt =
      static_cast<const clang::LabelStmt &>(stmt);

    exprt sub_stmt;
    if(get_expr(*label_stmt.getSubStmt(), sub_stmt))
      return true;

    convert_expression_to_code(sub_stmt);

    code_labelt label;
    label.set_label(label_stmt.getName());
    label.code() = to_code(sub_stmt);

    new_expr = label;
    break;
  }

  // An if then else statement. The else statement may not
  // exist, so we must check before constructing its exprt.
  // We always to try to cast its condition to bool
  case clang::Stmt::IfStmtClass:
  {
    const clang::IfStmt &ifstmt = static_cast<const clang::IfStmt &>(stmt);

    const clang::Stmt *cond_expr = ifstmt.getConditionVariableDeclStmt();
    if(cond_expr == nullptr)
      cond_expr = ifstmt.getCond();

    exprt cond;
    if(get_expr(*cond_expr, cond))
      return true;

    exprt then;
    if(get_expr(*ifstmt.getThen(), then))
      return true;

    convert_expression_to_code(then);

    codet if_expr("ifthenelse");
    if_expr.copy_to_operands(cond, then);

    const clang::Stmt *else_stmt = ifstmt.getElse();

    if(else_stmt)
    {
      exprt else_expr;
      if(get_expr(*else_stmt, else_expr))
        return true;

      convert_expression_to_code(else_expr);
      if_expr.copy_to_operands(else_expr);
    }

    new_expr = if_expr;
    break;
  }

  // A switch statement.
  case clang::Stmt::SwitchStmtClass:
  {
    const clang::SwitchStmt &switch_stmt =
      static_cast<const clang::SwitchStmt &>(stmt);

    const clang::Stmt *cond_expr = switch_stmt.getConditionVariableDeclStmt();
    if(cond_expr == nullptr)
      cond_expr = switch_stmt.getCond();

    exprt cond;
    if(get_expr(*cond_expr, cond))
      return true;

    codet body;
    if(get_expr(*switch_stmt.getBody(), body))
      return true;

    code_switcht switch_code;
    switch_code.value() = cond;
    switch_code.body() = body;

    new_expr = switch_code;
    break;
  }

  // A while statement. Even if its body is empty, an CompoundStmt
  // is generated for it. We always try to cast its condition to bool
  case clang::Stmt::WhileStmtClass:
  {
    const clang::WhileStmt &while_stmt =
      static_cast<const clang::WhileStmt &>(stmt);

    const clang::Stmt *cond_expr = while_stmt.getConditionVariableDeclStmt();
    if(cond_expr == nullptr)
      cond_expr = while_stmt.getCond();

    exprt cond;
    if(get_expr(*cond_expr, cond))
      return true;

    codet body = code_skipt();
    if(get_expr(*while_stmt.getBody(), body))
      return true;

    convert_expression_to_code(body);

    code_whilet code_while;
    code_while.cond() = cond;
    code_while.body() = body;

    new_expr = code_while;
    break;
  }

  // A dowhile statement. Even if its body is empty, an CompoundStmt
  // is generated for it. We always try to cast its condition to bool
  case clang::Stmt::DoStmtClass:
  {
    const clang::DoStmt &do_stmt = static_cast<const clang::DoStmt &>(stmt);

    exprt cond;
    if(get_expr(*do_stmt.getCond(), cond))
      return true;

    codet body = code_skipt();
    if(get_expr(*do_stmt.getBody(), body))
      return true;

    convert_expression_to_code(body);

    code_dowhilet code_while;
    code_while.cond() = cond;
    code_while.body() = body;

    new_expr = code_while;
    break;
  }

  // A For statement. Even if its body is empty, an CompoundStmt
  // is generated for it. We always try to cast its condition to bool.
  // Its parameters might be empty, so we have to check them all before
  // converting
  case clang::Stmt::ForStmtClass:
  {
    const clang::ForStmt &for_stmt = static_cast<const clang::ForStmt &>(stmt);

    codet init = code_skipt();
    const clang::Stmt *init_stmt = for_stmt.getInit();
    if(init_stmt)
      if(get_expr(*init_stmt, init))
        return true;

    convert_expression_to_code(init);
    const clang::Stmt *cond_expr = for_stmt.getConditionVariableDeclStmt();
    if(cond_expr == nullptr)
      cond_expr = for_stmt.getCond();

    exprt cond = true_exprt();
    if(cond_expr)
      if(get_expr(*cond_expr, cond))
        return true;

    codet inc = code_skipt();
    const clang::Stmt *inc_stmt = for_stmt.getInc();
    if(inc_stmt)
      get_expr(*inc_stmt, inc);

    convert_expression_to_code(inc);

    codet body = code_skipt();
    const clang::Stmt *body_stmt = for_stmt.getBody();
    if(body_stmt)
      if(get_expr(*body_stmt, body))
        return true;

    convert_expression_to_code(body);

    code_fort code_for;
    code_for.init() = init;
    code_for.cond() = cond;
    code_for.iter() = inc;
    code_for.body() = body;

    new_expr = code_for;
    break;
  }

  // a goto instruction to a label
  case clang::Stmt::GotoStmtClass:
  {
    const clang::GotoStmt &goto_stmt =
      static_cast<const clang::GotoStmt &>(stmt);

    code_gotot code_goto;
    code_goto.set_destination(goto_stmt.getLabel()->getName().str());

    new_expr = code_goto;
    break;
  }

  case clang::Stmt::IndirectGotoStmtClass:
  {
    const clang::IndirectGotoStmt &goto_stmt =
      static_cast<const clang::IndirectGotoStmt &>(stmt);

    // clang was able to compute the target, so this became a
    // common goto
    if(goto_stmt.getConstantTarget())
    {
      code_gotot code_goto;
      code_goto.set_destination(goto_stmt.getConstantTarget()->getName().str());

      new_expr = code_goto;
    }
    else
    {
      std::cerr << "ESBMC currently does not support indirect gotos"
                << std::endl;
      stmt.dumpColor();
      return true;

      exprt target;
      if(get_expr(*goto_stmt.getTarget(), target))
        return true;

      codet code_goto("gcc_goto");
      code_goto.copy_to_operands(target);

      new_expr = code_goto;
    }

    break;
  }

  // A continue statement
  case clang::Stmt::ContinueStmtClass:
    new_expr = code_continuet();
    break;

  // A break statement
  case clang::Stmt::BreakStmtClass:
    new_expr = code_breakt();
    break;

  // A return statement
  case clang::Stmt::ReturnStmtClass:
  {
    const clang::ReturnStmt &ret = static_cast<const clang::ReturnStmt &>(stmt);

    if(!current_functionDecl)
    {
      std::cerr << "ESBMC could not find the parent scope for "
                << "the following return statement:" << std::endl;
      ret.dumpColor();
      return true;
    }

    typet return_type;
    if(get_type(current_functionDecl->getReturnType(), return_type))
      return true;

    code_returnt ret_expr;
    if(ret.getRetValue())
    {
      const clang::Expr &retval = *ret.getRetValue();

      exprt val;
      if(get_expr(retval, val))
        return true;

      gen_typecast(ns, val, return_type);
      ret_expr.return_value() = val;
    }

    new_expr = ret_expr;
    break;
  }

  // Atomic instructions
  case clang::Stmt::AtomicExprClass:
  {
    const clang::AtomicExpr &atm = static_cast<const clang::AtomicExpr &>(stmt);

    if(get_atomic_expr(atm, new_expr))
      return true;

    break;
  }

  // A NULL statement, we ignore it. An example is a lost semicolon on
  // the program
  case clang::Stmt::NullStmtClass:

  // GCC or MS Assembly instruction. We ignore them
  case clang::Stmt::GCCAsmStmtClass:
  case clang::Stmt::MSAsmStmtClass:
    new_expr = code_skipt();
    break;

  default:
    std::cerr << "Conversion of unsupported clang expr: \"";
    std::cerr << stmt.getStmtClassName() << "\" to expression" << std::endl;
    stmt.dumpColor();
    return true;
  }

  new_expr.location() = location;
  return false;
}

bool clang_c_convertert::get_decl_ref(const clang::Decl &d, exprt &new_expr)
{
  // Special case for Enums, we return the constant instead of a reference
  // to the name
  if(const auto *e = llvm::dyn_cast<clang::EnumConstantDecl>(&d))
  {
    // For enum constants, we get their value directly
    new_expr = constant_exprt(
      integer2binary(e->getInitVal().getSExtValue(), bv_width(int_type())),
      integer2string(e->getInitVal().getSExtValue()),
      int_type());

    return false;
  }

  if(const auto *nd = llvm::dyn_cast<clang::ValueDecl>(&d))
  {
    // Everything else should be a value decl
    std::string name, id;
    get_decl_name(*nd, name, id);

    typet type;
    if(get_type(nd->getType(), type))
      return true;

    new_expr = exprt("symbol", type);
    new_expr.identifier(id);
    new_expr.cmt_lvalue(true);
    new_expr.name(name);
    return false;
  }

  std::cerr << "Conversion of unsupported clang decl ref: \"";
  std::cerr << d.getDeclKindName() << "\" to expression" << std::endl;
  d.dumpColor();
  return true;
}

bool clang_c_convertert::get_cast_expr(
  const clang::CastExpr &cast,
  exprt &new_expr)
{
  exprt expr;
  if(get_expr(*cast.getSubExpr(), expr))
    return true;

  typet type;
  if(get_type(cast.getType(), type))
    return true;

  switch(cast.getCastKind())
  {
  case clang::CK_ArrayToPointerDecay:
  case clang::CK_FunctionToPointerDecay:
  case clang::CK_BuiltinFnToFnPtr:
    break;

  case clang::CK_NoOp:

  case clang::CK_IntegralCast:
  case clang::CK_IntegralToBoolean:
  case clang::CK_IntegralToFloating:
  case clang::CK_IntegralToPointer:

  case clang::CK_FloatingToIntegral:
  case clang::CK_FloatingToBoolean:
  case clang::CK_FloatingCast:

  case clang::CK_ToVoid:
  case clang::CK_BitCast:
  case clang::CK_LValueToRValue:

  case clang::CK_PointerToBoolean:
  case clang::CK_PointerToIntegral:
    gen_typecast(ns, expr, type);
    break;

  case clang::CK_NullToPointer:
    expr = gen_zero(type);
    break;

  default:
    std::cerr << "Conversion of unsupported clang cast operator: \"";
    std::cerr << cast.getCastKindName() << "\" to expression" << std::endl;
    cast.dumpColor();
    return true;
  }

  new_expr = expr;
  return false;
}

bool clang_c_convertert::get_unary_operator_expr(
  const clang::UnaryOperator &uniop,
  exprt &new_expr)
{
  typet uniop_type;
  if(get_type(uniop.getType(), uniop_type))
    return true;

  exprt unary_sub;
  if(get_expr(*uniop.getSubExpr(), unary_sub))
    return true;

  switch(uniop.getOpcode())
  {
  case clang::UO_Plus:
    new_expr = exprt("unary+", uniop_type);
    break;

  case clang::UO_Minus:
    new_expr = exprt("unary-", uniop_type);
    break;

  case clang::UO_Not:
    new_expr = exprt("bitnot", uniop_type);
    break;

  case clang::UO_LNot:
    new_expr = exprt("not", bool_type());
    break;

  case clang::UO_PreInc:
    new_expr = side_effect_exprt("preincrement", uniop_type);
    break;

  case clang::UO_PreDec:
    new_expr = side_effect_exprt("predecrement", uniop_type);
    break;

  case clang::UO_PostInc:
    new_expr = side_effect_exprt("postincrement", uniop_type);
    break;

  case clang::UO_PostDec:
    new_expr = side_effect_exprt("postdecrement", uniop_type);
    break;

  case clang::UO_AddrOf:
    new_expr = exprt("address_of", uniop_type);
    break;

  case clang::UO_Deref:
    new_expr = exprt("dereference", uniop_type);
    break;

  case clang::UO_Extension:
    new_expr.swap(unary_sub);
    return false;

  default:
    std::cerr << "Conversion of unsupported clang unary operator: \"";
    std::cerr << clang::UnaryOperator::getOpcodeStr(uniop.getOpcode()).str()
              << "\" to expression" << std::endl;
    uniop.dumpColor();
    return true;
  }

  new_expr.operands().push_back(unary_sub);
  return false;
}

bool clang_c_convertert::get_binary_operator_expr(
  const clang::BinaryOperator &binop,
  exprt &new_expr)
{
  exprt lhs;
  if(get_expr(*binop.getLHS(), lhs))
    return true;

  exprt rhs;
  if(get_expr(*binop.getRHS(), rhs))
    return true;

  typet t;
  if(get_type(binop.getType(), t))
    return true;

  switch(binop.getOpcode())
  {
  case clang::BO_Add:
    if(t.is_floatbv())
      new_expr = exprt("ieee_add", t);
    else
      new_expr = exprt("+", t);
    break;

  case clang::BO_Sub:
    if(t.is_floatbv())
      new_expr = exprt("ieee_sub", t);
    else
      new_expr = exprt("-", t);
    break;

  case clang::BO_Mul:
    if(t.is_floatbv())
      new_expr = exprt("ieee_mul", t);
    else
      new_expr = exprt("*", t);
    break;

  case clang::BO_Div:
    if(t.is_floatbv())
      new_expr = exprt("ieee_div", t);
    else
      new_expr = exprt("/", t);
    break;

  case clang::BO_Shl:
    new_expr = exprt("shl", t);
    break;

  case clang::BO_Shr:
    new_expr = exprt("shr", t);
    break;

  case clang::BO_Rem:
    new_expr = exprt("mod", t);
    break;

  case clang::BO_And:
    new_expr = exprt("bitand", t);
    break;

  case clang::BO_Xor:
    new_expr = exprt("bitxor", t);
    break;

  case clang::BO_Or:
    new_expr = exprt("bitor", t);
    break;

  case clang::BO_LT:
    new_expr = exprt("<", t);
    break;

  case clang::BO_GT:
    new_expr = exprt(">", t);
    break;

  case clang::BO_LE:
    new_expr = exprt("<=", t);
    break;

  case clang::BO_GE:
    new_expr = exprt(">=", t);
    break;

  case clang::BO_EQ:
    new_expr = exprt("=", t);
    break;

  case clang::BO_NE:
    new_expr = exprt("notequal", t);
    break;

  case clang::BO_LAnd:
    new_expr = exprt("and", t);
    break;

  case clang::BO_LOr:
    new_expr = exprt("or", t);
    break;

  case clang::BO_Assign:
    // If we use code_assignt, it will reserve two operands,
    // and the copy_to_operands method call at the end of
    // this method will put lhs and rhs in positions 2 and 3,
    // instead of 0 and 1 :/
    new_expr = side_effect_exprt("assign", t);
    break;

  case clang::BO_Comma:
    new_expr = exprt("comma", t);
    break;

  default:
  {
    const clang::CompoundAssignOperator &compop =
      static_cast<const clang::CompoundAssignOperator &>(binop);
    return get_compound_assign_expr(compop, new_expr);
  }
  }

  new_expr.copy_to_operands(lhs, rhs);
  return false;
}

bool clang_c_convertert::get_compound_assign_expr(
  const clang::CompoundAssignOperator &compop,
  exprt &new_expr)
{
  switch(compop.getOpcode())
  {
  case clang::BO_AddAssign:
    new_expr = side_effect_exprt("assign+");
    break;

  case clang::BO_SubAssign:
    new_expr = side_effect_exprt("assign-");
    break;

  case clang::BO_MulAssign:
    new_expr = side_effect_exprt("assign*");
    break;

  case clang::BO_DivAssign:
    new_expr = side_effect_exprt("assign_div");
    break;

  case clang::BO_RemAssign:
    new_expr = side_effect_exprt("assign_mod");
    break;

  case clang::BO_ShlAssign:
    new_expr = side_effect_exprt("assign_shl");
    break;

  case clang::BO_ShrAssign:
    new_expr = side_effect_exprt("assign_shr");
    break;

  case clang::BO_AndAssign:
    new_expr = side_effect_exprt("assign_bitand");
    break;

  case clang::BO_XorAssign:
    new_expr = side_effect_exprt("assign_bitxor");
    break;

  case clang::BO_OrAssign:
    new_expr = side_effect_exprt("assign_bitor");
    break;

  default:
    std::cerr << "Conversion of unsupported clang binary operator: \"";
    std::cerr << compop.getOpcodeStr().str() << "\" to expression" << std::endl;
    compop.dumpColor();
    return true;
  }

  exprt lhs;
  if(get_expr(*compop.getLHS(), lhs))
    return true;

  exprt rhs;
  if(get_expr(*compop.getRHS(), rhs))
    return true;

  if(get_type(compop.getType(), new_expr.type()))
    return true;

  if(!lhs.type().is_pointer())
    gen_typecast(ns, rhs, lhs.type());

  new_expr.copy_to_operands(lhs, rhs);
  return false;
}

bool clang_c_convertert::get_atomic_expr(
  const clang::AtomicExpr &atm,
  exprt &new_expr)
{
  // We're going to create a fake function call
  side_effect_expr_function_callt fake_call;

  // Get the type
  code_typet t;
  if(get_type(atm.getType(), t.return_type()))
    return true;
  fake_call.type() = t;

  // get the name
  std::string name;
  switch(atm.getOp())
  {
  case clang::AtomicExpr::AO__c11_atomic_load:
    name = "__c11_atomic_load";
    break;

  case clang::AtomicExpr::AO__c11_atomic_store:
    name = "__c11_atomic_store";
    break;

  case clang::AtomicExpr::AO__c11_atomic_exchange:
    name = "__c11_atomic_exchange";
    break;

  case clang::AtomicExpr::AO__c11_atomic_compare_exchange_strong:
    name = "__c11_atomic_compare_exchange_strong";
    break;

  case clang::AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    name = "__c11_atomic_compare_exchange_weak";
    break;

  case clang::AtomicExpr::AO__c11_atomic_fetch_add:
    name = "__c11_atomic_fetch_add";
    break;

  case clang::AtomicExpr::AO__c11_atomic_fetch_sub:
    name = "__c11_atomic_fetch_sub";
    break;

  case clang::AtomicExpr::AO__c11_atomic_fetch_and:
    name = "__c11_atomic_fetch_and";
    break;

  case clang::AtomicExpr::AO__c11_atomic_fetch_or:
    name = "__c11_atomic_fetch_or";
    break;

  case clang::AtomicExpr::AO__c11_atomic_fetch_xor:
    name = "__c11_atomic_fetch_xor";
    break;

  case clang::AtomicExpr::AO__atomic_load:
    name = "__atomic_load";
    break;

  case clang::AtomicExpr::AO__atomic_load_n:
    name = "__atomic_load_n";
    break;

  case clang::AtomicExpr::AO__atomic_store:
    name = "__atomic_store";
    break;

  case clang::AtomicExpr::AO__atomic_store_n:
    name = "__atomic_store_n";
    break;

  case clang::AtomicExpr::AO__atomic_exchange:
    name = "__atomic_exchange";
    break;

  case clang::AtomicExpr::AO__atomic_exchange_n:
    name = "__atomic_exchange_n";
    break;

  case clang::AtomicExpr::AO__atomic_compare_exchange:
    name = "__atomic_compare_exchange";
    break;

  case clang::AtomicExpr::AO__atomic_compare_exchange_n:
    name = "__atomic_compare_exchange_n";
    break;

  case clang::AtomicExpr::AO__atomic_fetch_add:
    name = "__atomic_fetch_add";
    break;

  case clang::AtomicExpr::AO__atomic_fetch_sub:
    name = "__atomic_fetch_sub";
    break;

  case clang::AtomicExpr::AO__atomic_fetch_and:
    name = "__atomic_fetch_and";
    break;

  case clang::AtomicExpr::AO__atomic_fetch_or:
    name = "__atomic_fetch_or";
    break;

  case clang::AtomicExpr::AO__atomic_fetch_xor:
    name = "__atomic_fetch_xor";
    break;

  case clang::AtomicExpr::AO__atomic_fetch_nand:
    name = "__atomic_fetch_nand";
    break;

  case clang::AtomicExpr::AO__atomic_add_fetch:
    name = "__atomic_add_fetch";
    break;

  case clang::AtomicExpr::AO__atomic_sub_fetch:
    name = "__atomic_sub_fetch";
    break;

  case clang::AtomicExpr::AO__atomic_and_fetch:
    name = "__atomic_and_fetch";
    break;

  case clang::AtomicExpr::AO__atomic_or_fetch:
    name = "__atomic_or_fetch";
    break;

  case clang::AtomicExpr::AO__atomic_xor_fetch:
    name = "__atomic_xor_fetch";
    break;

  case clang::AtomicExpr::AO__atomic_nand_fetch:
    name = "__atomic_nand_fetch";
    break;

  default:
    std::cerr << "Unknown Atomic expression" << std::endl;
    atm.dumpColor();
    return true;
  }

  // Get Arguments, ptr is never nullptr
  exprt ptr;
  if(get_expr(*atm.getPtr(), ptr))
    return true;

  t.arguments().push_back(code_typet::argumentt(ptr.type()));
  fake_call.arguments().push_back(ptr);

  // Val1
  if(
    atm.getOp() != clang::AtomicExpr::AO__c11_atomic_load &&
    atm.getOp() != clang::AtomicExpr::AO__atomic_load_n)
  {
    exprt val1;
    if(get_expr(*atm.getVal1(), val1))
      return true;

    t.arguments().push_back(code_typet::argumentt(val1.type()));
    fake_call.arguments().push_back(val1);
  }

  // Val2
  if(atm.getOp() == clang::AtomicExpr::AO__atomic_exchange || atm.isCmpXChg())
  {
    exprt val2;
    if(get_expr(*atm.getVal2(), val2))
      return true;

    t.arguments().push_back(code_typet::argumentt(val2.type()));
    fake_call.arguments().push_back(val2);
  }

  // Weak
  if(
    atm.getOp() == clang::AtomicExpr::AO__atomic_compare_exchange ||
    atm.getOp() == clang::AtomicExpr::AO__atomic_compare_exchange_n)
  {
    exprt weak;
    if(get_expr(*atm.getWeak(), weak))
      return true;

    t.arguments().push_back(code_typet::argumentt(weak.type()));
    fake_call.arguments().push_back(weak);
  }

  if(atm.getOp() != clang::AtomicExpr::AO__c11_atomic_init)
  {
    exprt order;
    if(get_expr(*atm.getOrder(), order))
      return true;

    t.arguments().push_back(code_typet::argumentt(order.type()));
    fake_call.arguments().push_back(order);
  }

  if(atm.isCmpXChg())
  {
    exprt order_fail;
    if(get_expr(*atm.getOrderFail(), order_fail))
      return true;

    t.arguments().push_back(code_typet::argumentt(order_fail.type()));
    fake_call.arguments().push_back(order_fail);
  }

  fake_call.function() = symbol_exprt("c:@F@" + name, t);
  fake_call.function().name(name);
  new_expr.swap(fake_call);
  return false;
}

void clang_c_convertert::get_default_symbol(
  symbolt &symbol,
  std::string module_name,
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbol.mode = "C";
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

static std::string get_decl_name(const clang::NamedDecl &nd)
{
  if(const clang::IdentifierInfo *identifier = nd.getIdentifier())
    return identifier->getName().str();

  std::string name;
  llvm::raw_string_ostream rso(name);
  nd.printName(rso);
  return rso.str();
}

static std::string
getFullyQualifiedName(const clang::QualType &t, const clang::ASTContext &c)
{
  clang::PrintingPolicy Policy(c.getPrintingPolicy());
  Policy.SuppressScope = false;
  Policy.AnonymousTagLocations = true;
  Policy.PolishForDeclaration = true;
  Policy.SuppressUnwrittenScope = true;
  return clang::TypeName::getFullyQualifiedName(t, c, Policy);
}

void clang_c_convertert::get_decl_name(
  const clang::NamedDecl &nd,
  std::string &name,
  std::string &id)
{
  id = name = ::get_decl_name(nd);

  switch(nd.getKind())
  {
  // ParamVarDecl, we can safely ignore them
  case clang::Decl::ParmVar:
    if(name.empty())
      return;
    break;

  case clang::Decl::Field:
    if(name.empty())
    {
      // Anonymous fields, generate a name based on the type
      const clang::FieldDecl &fd = static_cast<const clang::FieldDecl &>(nd);
      name = "anon";
      id = getFullyQualifiedName(fd.getType(), *ASTContext);
    }
    return;

  case clang::Decl::IndirectField:
    if(name.empty())
    {
      // Anonymous fields, generate a name based on the type
      const clang::IndirectFieldDecl &fd =
        static_cast<const clang::IndirectFieldDecl &>(nd);
      name = "anon";
      id = getFullyQualifiedName(fd.getType(), *ASTContext);
      return;
    }
    break;

  case clang::Decl::Record:
  {
    const clang::RecordDecl &rd = static_cast<const clang::RecordDecl &>(nd);
    name = getFullyQualifiedName(ASTContext->getTagDeclType(&rd), *ASTContext);
    id = "tag-" + name;
    return;
  }

  default:
    if(name.empty())
    {
      std::cerr << "Declaration has an empty name:\n";
      nd.dumpColor();
      abort();
    }
  }

  clang::SmallString<128> DeclUSR;
  if(!clang::index::generateUSRForDecl(&nd, DeclUSR))
  {
    id = DeclUSR.str().str();
    return;
  }

  // Otherwise, abort
  std::cerr << "Unable to generate the USR for:\n";
  nd.dumpColor();
  abort();
}

void clang_c_convertert::get_start_location_from_stmt(
  const clang::Stmt &stmt,
  locationt &location)
{
  sm = &ASTContext->getSourceManager();

  std::string function_name;

  if(current_functionDecl)
    function_name = current_functionDecl->getName().str();

  clang::PresumedLoc PLoc;
  get_presumed_location(stmt.getSourceRange().getBegin(), PLoc);

  set_location(PLoc, function_name, location);
}

void clang_c_convertert::get_final_location_from_stmt(
  const clang::Stmt &stmt,
  locationt &location)
{
  sm = &ASTContext->getSourceManager();

  std::string function_name;

  if(current_functionDecl)
    function_name = current_functionDecl->getName().str();

  clang::PresumedLoc PLoc;
  get_presumed_location(stmt.getSourceRange().getEnd(), PLoc);

  set_location(PLoc, function_name, location);
}

void clang_c_convertert::get_location_from_decl(
  const clang::Decl &decl,
  locationt &location)
{
  sm = &ASTContext->getSourceManager();

  std::string function_name;

  if(decl.getDeclContext()->isFunctionOrMethod())
  {
    const clang::FunctionDecl &funcd =
      static_cast<const clang::FunctionDecl &>(*decl.getDeclContext());

    function_name = funcd.getName().str();
  }

  clang::PresumedLoc PLoc;
  get_presumed_location(decl.getSourceRange().getBegin(), PLoc);

  set_location(PLoc, function_name, location);
}

void clang_c_convertert::get_presumed_location(
  const clang::SourceLocation &loc,
  clang::PresumedLoc &PLoc)
{
  if(!sm)
    return;

  clang::SourceLocation SpellingLoc = sm->getSpellingLoc(loc);
  PLoc = sm->getPresumedLoc(SpellingLoc);
}

void clang_c_convertert::set_location(
  clang::PresumedLoc &PLoc,
  std::string &function_name,
  locationt &location)
{
  if(PLoc.isInvalid())
  {
    location.set_file("<invalid sloc>");
    return;
  }

  location.set_line(PLoc.getLine());
  location.set_file(get_filename_from_path(PLoc.getFilename()));

  if(!function_name.empty())
    location.set_function(function_name);
}

std::string clang_c_convertert::get_modulename_from_path(std::string path)
{
  std::string filename = get_filename_from_path(path);

  if(filename.find_last_of('.') != std::string::npos)
    return filename.substr(0, filename.find_last_of('.'));

  return filename;
}

std::string clang_c_convertert::get_filename_from_path(std::string path)
{
  if(path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path;
}

symbolt *clang_c_convertert::move_symbol_to_context(symbolt &symbol)
{
  symbolt *s = context.find_symbol(symbol.id);
  if(s == nullptr)
  {
    if(context.move(symbol, s))
    {
      std::cerr << "Couldn't add symbol " << symbol.name
                << " to symbol table\n";
      symbol.dump();
      abort();
    }
  }
  else
  {
    // types that are code means functions
    if(s->type.is_code())
    {
      if(symbol.value.is_not_nil() && !s->value.is_not_nil())
        s->swap(symbol);
    }
    else if(s->is_type)
    {
      if(symbol.type.is_not_nil() && !s->type.is_not_nil())
        s->swap(symbol);
    }
  }

  return s;
}

void clang_c_convertert::dump_type_map()
{
  std::cout << "Type_map:" << std::endl;
  for(auto const &it : type_map)
    std::cout << it.first << ": " << it.second << std::endl;
}

void clang_c_convertert::convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

bool clang_c_convertert::search_add_type_map(
  const clang::TagDecl &tag,
  type_mapt::iterator &type_it)
{
  std::size_t address = reinterpret_cast<std::size_t>(tag.getFirstDecl());

  // Search for the type on the type map
  type_it = type_map.find(address);
  if(type_it == type_map.end())
  {
    // Force the declaration to be added to the type_map
    exprt decl;
    if(get_decl(tag, decl))
      return true;
  }

  type_it = type_map.find(address);
  if(type_it == type_map.end())
    // BUG! This should be added already
    return true;

  return false;
}

const clang::Decl *
clang_c_convertert::get_DeclContext_from_Stmt(const clang::Stmt &stmt)
{
  auto it = ASTContext->getParents(stmt).begin();
  if(it == ASTContext->getParents(stmt).end())
    return nullptr;

  const clang::Decl *aDecl = it->get<clang::Decl>();
  if(aDecl)
    return aDecl;

  const clang::Stmt *aStmt = it->get<clang::Stmt>();
  if(aStmt)
    return get_DeclContext_from_Stmt(*aStmt);

  return nullptr;
}

const clang::Decl *
clang_c_convertert::get_top_FunctionDecl_from_Stmt(const clang::Stmt &stmt)
{
  const clang::Decl *decl = get_DeclContext_from_Stmt(stmt);
  if(decl)
  {
    if(decl->isFunctionOrFunctionTemplate())
      return decl;

    if(decl->getNonClosureContext()->isFunctionOrFunctionTemplate())
      return decl->getNonClosureContext();
  }

  return nullptr;
}

bool clang_c_convertert::has_bitfields(const typet &_type, typet *converted)
{
  typet type = _type;
  if(type.id() == "symbol")
    type = ns.follow(type);

  auto fixed_it = bitfield_fixed_type_map.find(type);
  if(fixed_it != bitfield_fixed_type_map.end())
  {
    if(converted)
      *converted = fixed_it->second;
    return true; // Yes, and the unfixed version version was passed in
  }

  auto orig_it = bitfield_orig_type_map.find(type);
  if(orig_it != bitfield_orig_type_map.end())
  {
    if(converted)
      *converted = type;
    return true; // Yes, and this is the fixed version
  }

  if(type.id() != "struct" && type.id() != "union")
    return false; // Could have been a symbol of a union

  auto sutype = to_struct_union_type(type);
  for(const auto &comp : sutype.components())
  {
    if(comp.type().get("#bitfield").as_string() == "true")
    {
      if(converted)
        *converted = typet();
      return true; // Yes, this is unconverted, and to date unknown.
    }
  }

  return false;
}

static std::string gen_bitfield_blob_name(unsigned int num)
{
  return "#BITFIELD" + std::to_string(num);
}

typet clang_c_convertert::fix_bitfields(const typet &_type)
{
  typet type = _type;
  if(type.id() == "symbol")
    type = ns.follow(type);

  if(bitfield_fixed_type_map.find(type) != bitfield_fixed_type_map.end())
    return bitfield_fixed_type_map.find(type)->second;

  typet new_type = type;
  auto sutype = to_struct_union_type(new_type);
  auto &components = sutype.components(); // It's a vector of components
  std::decay<decltype(components)>::type new_components;
  bool is_packed = sutype.get("packed").as_string() == "true";

  unsigned int bit_offs = 0;
  unsigned int blob_count = 0;

  std::map<irep_idt, bitfield_map> backmap;

  auto pop_blob = [is_packed, &bit_offs, &blob_count, &new_components]() {
    // We have to pop the current bitfield blob into the struct and create
    // a new one to make space.

    // Size of the bitfield depends on whether we're packed or not.
    typet ubv;
    if(is_packed)
    {
      // Round up to nearest byte.
      bit_offs += 7;
      bit_offs &= 0xFFFFFFF8;
      ubv = unsignedbv_typet(bit_offs);
    }
    else
    {
      // Always generate a 64 bit blob for now, optimise later.
      ubv = unsignedbv_typet(BITFIELD_MAX_FIELD);
    }

    std::string name = gen_bitfield_blob_name(blob_count);
    struct_union_typet::componentt newcomp(name, name, ubv);
    new_components.push_back(newcomp);

    bit_offs = 0;
    blob_count++;
  };

  for(const auto &comp : components)
  { // Go through all components...
    if(comp.type().get("#bitfield").as_string() != "true")
    {
      if(bit_offs != 0)
        pop_blob();

      new_components.push_back(comp);
      continue;
    }

    // Otherwise: this is a bitfield.
    unsigned int width = bv_width(comp.type());
    assert(width <= BITFIELD_MAX_FIELD && "Humoungous bitfield");

    if(width + bit_offs > BITFIELD_MAX_FIELD)
      pop_blob();

    // Add this bitfield to the current blob.
    backmap.insert(
      std::make_pair(comp.name(), bitfield_map{bit_offs, blob_count}));
    bit_offs += width;
  }

  if(bit_offs != 0)
    pop_blob();

  // We now have: to replace the components in the new type, and store the
  // backmap so that subsequent read/writes can work out what replacement
  // operation to build.
  sutype.components() = new_components;
  bitfield_mappings.insert(std::make_pair(sutype, std::move(backmap)));
  bitfield_fixed_type_map.insert(std::make_pair(type, sutype));
  bitfield_orig_type_map.insert(std::make_pair(sutype, type));

  return sutype;
}

void clang_c_convertert::fix_constant_bitfields(exprt &expr)
{
  assert(expr.type().id() == "struct");
  assert(
    bitfield_fixed_type_map.find(expr.type()) != bitfield_fixed_type_map.end());

  // Well, we now need to fix up this constant struct expr into one that doesn't
  // feature any bitfields, because:
  auto sutype = to_struct_union_type(bitfield_fixed_type_map[expr.type()]);
  bool is_packed = sutype.get("packed").as_string() == "true";

  const auto &orig_type = to_struct_union_type(expr.type());

  exprt new_expr = expr;
  new_expr.operands().clear();
  new_expr.type() = sutype;
  exprt accuml;
  accuml.make_nil();

  unsigned int bit_offs = 0;

  auto pop_blob = [is_packed, &accuml, &bit_offs, &new_expr]() {
    if(is_packed)
    {
      // Round number of bits up to nearest byte,
      bit_offs += 7;
      bit_offs &= 0xFFFFFFF8;
      accuml = typecast_exprt(accuml, unsignedbv_typet(bit_offs));
    }
    else if(bv_width(accuml.type()) != BITFIELD_MAX_FIELD)
    {
      accuml = typecast_exprt(accuml, unsignedbv_typet(BITFIELD_MAX_FIELD));
    } // Otherwise it's already at the right size

    new_expr.operands().push_back(accuml);
    accuml = exprt();
    accuml.make_nil();
    bit_offs = 0;
  };

  // OK: iterate through all operands, concatting the bitfields, and then
  // storing back into the operand list. XXX, work out how to use bitfield
  // map to make this better?
  for(unsigned int i = 0; i < expr.operands().size(); ++i)
  {
    const exprt &orig_elem = expr.operands()[i];
    const exprt &orig_comp = orig_type.components()[i];
    if(orig_comp.type().get("#bitfield") != "true")
    {
      if(bit_offs > 0)
        pop_blob();

      new_expr.operands().push_back(orig_elem);
      continue;
    }

    unsigned int width = bv_width(orig_elem.type());
    if(bit_offs + width > BITFIELD_MAX_FIELD)
      pop_blob();

    if(accuml.is_nil())
    {
      accuml = orig_elem;
    }
    else
    {
      unsigned int a_width = bv_width(accuml.type());
      unsigned int b_width = bv_width(orig_elem.type());
      unsignedbv_typet ubv(a_width + b_width);
      exprt tmp("concat", ubv);
      tmp.copy_to_operands(orig_elem, accuml); // XXX: ordering!
      accuml = tmp;
    }

    bit_offs += width;
  }

  if(bit_offs != 0)
    pop_blob();

  // We should now have replaced all operands corresponding to bitfields with
  // single concatted operands.
  assert(sutype.components().size() == new_expr.operands().size());
  // Just check the sub-operands now.
  expr = new_expr;
}

void clang_c_convertert::rewrite_bitfield_member(
  exprt &expr,
  const bitfield_map &bm)
{
  // The plan: build a new member expression accessing the blob field that
  // contains the bitfield. Then create an extract expression that pulls out
  // the relevant bits.

  // Erk. Clang applies members to pointers sometimes.
  auto &memb = to_member_expr(expr);
  typet basetype = expr.op0().type();
  if(basetype.id() == "pointer")
  {
    basetype = basetype.subtype();
    basetype = ns.follow(basetype);
  }

  auto &sutype = to_struct_union_type(basetype);

  std::string fieldname = gen_bitfield_blob_name(bm.blobloc);
  auto &this_comp = sutype.get_component(fieldname);
  assert(bv_width(this_comp.type()) != 0);
  unsignedbv_typet ubv_size(bv_width(this_comp.type()));
  // Note that we depend on the member expression still carrying the bitfield
  // width here. If that isn't true in the future, it'll have to be stored in
  // the bitfield map struct.
  unsigned int our_width = bv_width(memb.type());

  member_exprt new_memb(memb.struct_op(), irep_idt(fieldname), ubv_size);

  // Welp. Today, the bit ordering is that the 'bitloc' is the bottommost bit.
  exprt extract("extract", memb.type());
  extract.copy_to_operands(new_memb);
  extract.set("lower", irep_idt(std::to_string(bm.bitloc)));
  extract.set("upper", irep_idt(std::to_string(bm.bitloc + (our_width - 1))));
  expr = extract;
}
