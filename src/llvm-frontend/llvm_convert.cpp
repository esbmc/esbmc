/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_convert.h"

#include <std_code.h>
#include <std_expr.h>
#include <expr_util.h>
#include <mp_arith.h>
#include <arith_tools.h>
#include <i2string.h>
#include <bitvector.h>

#include <ansi-c/c_types.h>
#include <ansi-c/ansi_c_expr.h>
#include <ansi-c/type2name.h>

#include <boost/filesystem.hpp>

#include "typecast.h"

llvm_convertert::llvm_convertert(
  contextt &_context,
  std::vector<std::unique_ptr<clang::ASTUnit> > &_ASTs)
  : ASTContext(&(*(*_ASTs.begin())->top_level_begin())->getASTContext()),
    context(_context),
    ns(context),
    ASTs(_ASTs),
    current_path(""),
    current_scope_var_num(1),
    anon_counter(0),
    sm(nullptr)
{
}

llvm_convertert::~llvm_convertert()
{
}

bool llvm_convertert::convert()
{
  if(convert_builtin_types())
    return true;

  if(convert_top_level_decl())
    return true;

  return false;
}

bool llvm_convertert::convert_builtin_types()
{
  // Convert va_list_tag
  // TODO: from clang 3.8 we'll have a member VaListTagDecl and a method
  // getVaListTagDecl() that might make the following code redundant
  clang::QualType q_va_list_type = ASTContext->getVaListTagType();
  if(!q_va_list_type.isNull())
  {
    const clang::TypedefType &t =
      static_cast<const clang::TypedefType &>(*q_va_list_type.getTypePtr());

    exprt dummy;
    get_decl(*t.getDecl(), dummy);
  }

  // TODO: clang offers several informations from the target architecture,
  // such as primitive type's size, much like our configt class. We could
  // offer an option in the future to query them from the target system.
  // See clang/Basic/TargetInfo.h for what clang has to offer

  return false;
}

bool llvm_convertert::convert_top_level_decl()
{
  // Iterate through each translation unit and their global symbols, creating
  // symbols as we go.
  for (auto &translation_unit : ASTs)
  {
    for (clang::ASTUnit::top_level_iterator
      it = translation_unit->top_level_begin();
      it != translation_unit->top_level_end();
      it++)
    {
      // Update ASTContext as it changes for each source file
      ASTContext = &(*it)->getASTContext();

      exprt dummy_decl;
      get_decl(**it, dummy_decl);
    }
  }

  return false;
}

// This method convert declarations. They are called when those declarations
// are to be added to the context. If a variable or function is being called
// but then get_decl_expr is called instead
void llvm_convertert::get_decl(
  const clang::Decl& decl,
  exprt &new_expr)
{
  new_expr = code_skipt();

  switch (decl.getKind())
  {
    // Label declaration
    case clang::Decl::Label:
    {
      std::cerr << "ESBMC currently does not support label declaration"
                << std::endl;
      abort();

      const clang::LabelDecl &ld =
        static_cast<const clang::LabelDecl&>(decl);

      exprt label("label", empty_typet());
      label.identifier(ld.getName().str());
      label.cmt_base_name(ld.getName().str());

      new_expr = label;
      break;
    }

    // Declaration of variables
    case clang::Decl::Var:
    {
      const clang::VarDecl &vd =
        static_cast<const clang::VarDecl&>(decl);
      get_var(vd, new_expr);
      break;
    }

    // Declaration of function's parameter
    case clang::Decl::ParmVar:
    {
      const clang::ParmVarDecl &param =
        static_cast<const clang::ParmVarDecl &>(decl);
      get_function_params(param, new_expr);
      break;
    }

    // Declaration of functions
    case clang::Decl::Function:
    {
      const clang::FunctionDecl &fd =
        static_cast<const clang::FunctionDecl&>(decl);
      get_function(fd, new_expr);
      break;
    }

    // Field inside a struct/union
    case clang::Decl::Field:
    {
      const clang::FieldDecl &fd =
        static_cast<const clang::FieldDecl&>(decl);

      typet t;
      get_type(fd.getType(), t);

      std::string field_identifier =
        get_var_name(fd.getName().str(), "");

      struct_union_typet::componentt comp(field_identifier, t);
      comp.set_pretty_name(fd.getName().str());

      if(fd.isBitField())
      {
        exprt width;
        get_expr(*fd.getBitWidth(), width);
        comp.type().width(width.cformat());
      }

      new_expr.swap(comp);
      break;
    }

    // Typedef declaration
    case clang::Decl::Typedef:
    {
      const clang::TypedefDecl &tdd =
        static_cast<const clang::TypedefDecl&>(decl);
      get_typedef(tdd, new_expr);
      break;
    }

    case clang::Decl::IndirectField:
    {
      const clang::IndirectFieldDecl &fd =
        static_cast<const clang::IndirectFieldDecl &>(decl);

      typet t;
      get_type(fd.getType(), t);

      struct_union_typet::componentt comp(fd.getName().str(), t);
      comp.set_pretty_name(fd.getName().str());

      if(fd.getAnonField()->isBitField())
      {
        exprt width;
        get_expr(*fd.getAnonField()->getBitWidth(), width);
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
      get_struct_union_class(record, new_expr);
      break;
    }

    // This is an empty declaration. An lost semicolon on the
    // code is an empty declaration
    case clang::Decl::Empty:

    // If this fails, llvm will not generate the ASTs, we can
    // safely skip it
    case clang::Decl::StaticAssert:

    // Enum declaration and values, we can safely skip them as
    // any occurrence of those will be converted to int type (enum)
    // or integer value (enum constant)
    case clang::Decl::Enum:
    case clang::Decl::EnumConstant:
      break;

    case clang::Decl::Namespace:
    case clang::Decl::TypeAlias:
    case clang::Decl::FileScopeAsm:
    case clang::Decl::Block:
    case clang::Decl::Captured:
    case clang::Decl::Import:
    default:
      std::cerr << "Unrecognized / unimplemented decl "
                << decl.getDeclKindName() << std::endl;
      decl.dumpColor();
      abort();
  }
}

void llvm_convertert::get_struct_union_class(
  const clang::RecordDecl& recordd,
  exprt& new_expr)
{
  if(recordd.isClass())
  {
    std::cerr << "Class is not supported yet" << std::endl;
    abort();
  }
  else if(recordd.isInterface())
  {
    std::cerr << "Interface is not supported yet" << std::endl;
    abort();
  }

  struct_union_typet t;
  if(recordd.isStruct())
    t = struct_typet();
  else if(recordd.isUnion())
    t = union_typet();
  else
    // This should never be reached
    abort();

  // Try to get the definition
  clang::RecordDecl *record_def = recordd.getDefinition();

  std::string identifier = get_tag_name(recordd);
  t.tag(identifier);

  locationt location_begin;
  get_location_from_decl(recordd, location_begin);

  symbolt symbol;
  get_default_symbol(
    symbol,
    t,
    identifier,
    identifier,
    location_begin,
    false); // There is no such thing as static struct/union/class on ANSI-C

  // Save the struct/union/class type address and name to the type map
  std::string symbol_name = symbol.name.as_string();

  std::size_t address =
    reinterpret_cast<std::size_t>(recordd.getFirstDecl());
  type_map[address] = symbol_name;

  symbol.is_type = true;

  // We have to add the struct/union/class to the context before converting its fields
  // because there might be recursive struct/union/class (pointers) and the code at
  // get_type, case clang::Type::Record, needs to find the correct type
  // (itself). Note that the type is incomplete at this stage, it doesn't
  // contain the fields, which are added to the symbol later on this method.
  move_symbol_to_context(symbol);

  // Don't continue to parse if it doesn't have a complete definition
  if(!record_def)
    return;

  // Now get the symbol back to continue the conversion
  symbolt &added_symbol = context.symbols.find(symbol_name)->second;

  get_struct_union_class_fields(*record_def, t);
  added_symbol.type = t;

  // This change on the pretty_name is just to beautify the output
  if(recordd.isStruct())
    added_symbol.pretty_name = "struct " + identifier;
  else if(recordd.isUnion())
    added_symbol.pretty_name = "union " + identifier;
}

void llvm_convertert::get_struct_union_class_fields(
  const clang::RecordDecl &recordd,
  struct_union_typet &type)
{
  for(const auto &decl : recordd.decls())
  {
    struct_typet::componentt comp;
    get_decl(*decl, comp);

    // If we are parsing a field declaration, add it to the components
    if(decl->getKind() == clang::Decl::Field)
    {
      if(comp.type().get_bool("anonymous"))
      {
        comp.name(comp.type().tag());
        comp.pretty_name(comp.type().tag());
      }

      type.components().push_back(comp);
    }
  }
}

void llvm_convertert::get_typedef(
  const clang::TypedefDecl &tdd,
  exprt &new_expr)
{
  // Get type
  typet t;
  get_type(tdd.getUnderlyingType().getCanonicalType(), t);

  locationt location_begin;
  get_location_from_decl(tdd, location_begin);

  symbolt symbol;
  get_default_symbol(
    symbol,
    t,
    tdd.getName().str(),
    get_modulename_from_path() + "::" + tdd.getName().str(),
    location_begin,
    false); // There is no such thing as static typedef on ANSI-C

  symbol.is_type = true;
  symbol.is_macro = true;

  move_symbol_to_context(symbol);
}

void llvm_convertert::get_var(
  const clang::VarDecl &vd,
  exprt &new_expr)
{
  // Get type
  typet t;
  get_type(vd.getType(), t);

  std::string function_name = "";
  if(vd.getDeclContext()->isFunctionOrMethod())
  {
    const clang::FunctionDecl &funcd =
      static_cast<const clang::FunctionDecl &>(*vd.getDeclContext());

    function_name = funcd.getName().str();
  }

  std::string identifier = get_var_name(vd.getName().str(), function_name);

  locationt location_begin;
  get_location_from_decl(vd, location_begin);

  symbolt symbol;
  get_default_symbol(
    symbol,
    t,
    vd.getName().str(),
    identifier,
    location_begin,
    !vd.isExternallyVisible());

  if (vd.hasGlobalStorage() && !vd.hasInit())
  {
    // Initialize with zero value, if the symbol has initial value,
    // it will be add later on this method
    symbol.value = gen_zero(t, true);
    symbol.value.zero_initializer(true);
  }

  symbol.lvalue = true;
  symbol.static_lifetime = vd.hasGlobalStorage();
  symbol.is_extern = vd.hasExternalStorage();
  symbol.file_local = !vd.isExternallyVisible();

  // Save the variable address and name to the object map
  std::string symbol_name = symbol.name.as_string();

  std::size_t address = reinterpret_cast<std::size_t>(&vd);
  object_map[address] = symbol_name;

  // We have to add the symbol before converting the initial assignment
  // because we might have something like 'int x = x + 1;' which is
  // completely wrong but allowed by the language
  move_symbol_to_context(symbol);

  // Now get the symbol back to continue the conversion
  symbolt &added_symbol = context.symbols.find(symbol_name)->second;

  code_declt decl;
  decl.operands().push_back(symbol_expr(added_symbol));

  if(vd.hasInit())
  {
    exprt val;
    get_expr(*vd.getInit(), val);
    gen_typecast(ns, val, t);

    added_symbol.value = val;
    decl.operands().push_back(val);
  }

  decl.location() = location_begin;

  new_expr = decl;
}

void llvm_convertert::get_function(
  const clang::FunctionDecl &fd,
  exprt &new_expr)
{
  // If the function is not defined but this is not the definition, skip it
  if(fd.isDefined() && !fd.isThisDeclarationADefinition())
    return;

  // TODO: use fd.isMain to flag and check the flag on llvm_adjust_expr
  // to saner way to add argc/argv/envp

  // Set initial variable name, it will be used for variables' name
  // This will be reset every time a function is parsed
  current_scope_var_num = 1;

  // Build function's type
  code_typet type;

  // Return type
  typet return_type;
  get_type(fd.getReturnType(), return_type);
  type.return_type() = return_type;

  if(fd.isVariadic())
    type.make_ellipsis();

  if(fd.isInlined())
    type.inlined(true);

  locationt location_begin;
  get_location_from_decl(fd, location_begin);

  symbolt symbol;
  get_default_symbol(
    symbol,
    type,
    fd.getName().str(),
    fd.getName().str(),
    location_begin,
    !fd.isExternallyVisible());

  std::string symbol_name = symbol.name.as_string();

  // Save the function address and name to the object map
  std::size_t address = reinterpret_cast<std::size_t>(&fd);
  object_map[address] = symbol.name.as_string();

  symbol.lvalue = true;
  symbol.is_extern = fd.getStorageClass() == clang::SC_Extern
                     || fd.getStorageClass() == clang::SC_PrivateExtern;
  symbol.file_local = !fd.isExternallyVisible();

  move_symbol_to_context(symbol);

  // Now get the symbol back to continue the conversion
  symbolt &added_symbol = context.symbols.find(symbol_name)->second;

  // We convert the parameters first so their symbol are added to context
  // before converting the body, as they may appear on the function body
  for (const auto &pdecl : fd.params())
  {
    code_typet::argumentt param;
    get_function_params(*pdecl, param);
    type.arguments().push_back(param);
  }

  added_symbol.type = type;

  // We need: a type, a name, and an optional body
  clang::Stmt *body = fd.getBody();

  if(body)
  {
    exprt body_exprt;
    get_expr(*body, body_exprt);
    added_symbol.value = body_exprt;
  }
}

void llvm_convertert::get_function_params(
  const clang::ParmVarDecl &pdecl,
  exprt &param)
{
  std::string name = pdecl.getName().str();

  typet param_type;
  get_type(pdecl.getOriginalType(), param_type);

  param = code_typet::argumentt();
  param.type() = param_type;
  param.cmt_base_name(name);

  // If the name is empty, this is an function definition that we don't
  // need to worry about as the function params name's will be defined
  // when the function is defined, the exprt is filled for the sake of
  // beautification
  if(name.empty())
    return;

  locationt location_begin;
  get_location_from_decl(pdecl, location_begin);

  const clang::FunctionDecl &funcd =
    static_cast<const clang::FunctionDecl &>(*pdecl.getParentFunctionOrMethod());

  std::string function_name = funcd.getName().str();

  symbolt param_symbol;
  get_default_symbol(
    param_symbol,
    param_type,
    name,
    get_param_name(name, function_name),
    location_begin,
    false); // function parameter cannot be static

  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;

  param.cmt_identifier(param_symbol.name.as_string());
  param.location() = param_symbol.location;

  // Save the function's param address and name to the object map
  std::size_t address = reinterpret_cast<std::size_t>(&pdecl);
  object_map[address] = param_symbol.name.as_string();

  // If the function is not defined, we don't need to add it's parameter
  // to the context, they will never be used
  if(!funcd.isDefined())
    return;

  move_symbol_to_context(param_symbol);
}

void llvm_convertert::get_type(
  const clang::QualType &q_type,
  typet &new_type)
{
  const clang::Type &the_type = *q_type.getTypePtrOrNull();

  switch (the_type.getTypeClass())
  {
    // Builtin types like integer
    case clang::Type::Builtin:
    {
      const clang::BuiltinType &bt =
        static_cast<const clang::BuiltinType&>(the_type);
      get_builtin_type(bt, new_type);
      break;
    }

    // Types using parenthesis, e.g. int (a);
    case clang::Type::Paren:
    {
      const clang::ParenType &pt =
        static_cast<const clang::ParenType&>(the_type);
      get_type(pt.getInnerType(), new_type);
      break;
    }

    // Pointer types
    case clang::Type::Pointer:
    {
      const clang::PointerType &pt =
        static_cast<const clang::PointerType &>(the_type);
      const clang::QualType &pointee = pt.getPointeeType();

      typet sub_type;
      get_type(pointee, sub_type);

      // Special case, pointers to structs/unions/classes must not
      // have a copy of it, but a reference to the type
      // TODO: classes
      if(sub_type.is_struct() || sub_type.is_union())
      {
        struct_union_typet t = to_struct_union_type(sub_type);
        sub_type = symbol_typet("c::" + t.tag().as_string());
      }

      new_type = gen_pointer_type(sub_type);
      break;
    }

    // Types adjusted by the semantic engine
    case clang::Type::Decayed:
    {
      const clang::DecayedType &pt =
        static_cast<const clang::DecayedType&>(the_type);
      get_type(pt.getDecayedType(), new_type);
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
                      "than 64 bits" << std::endl;
        abort();
      }

      typet the_type;
      get_type(arr.getElementType(), the_type);

      array_typet type(the_type);
      type.size() =
        constant_exprt(
          integer2binary(val.getSExtValue(), bv_width(int_type())),
          integer2string(val.getSExtValue()),
          int_type());

      new_type = type;
      break;
    }

    // Array with undefined type, as in function args
    case clang::Type::IncompleteArray:
    {
      const clang::IncompleteArrayType &arr =
        static_cast<const clang::IncompleteArrayType &>(the_type);

      typet sub_type;
      get_type(arr.getElementType(), sub_type);

      new_type = gen_pointer_type(sub_type);
      break;
    }

    // Array with variable size, e.g., int a[n];
    case clang::Type::VariableArray:
    {
      const clang::VariableArrayType &arr =
        static_cast<const clang::VariableArrayType &>(the_type);

      exprt size_expr;
      get_expr(*arr.getSizeExpr(), size_expr);

      typet the_type;
      get_type(arr.getElementType(), the_type);

      array_typet type;
      type.size() = size_expr;
      type.subtype() = the_type;

      new_type = type;
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
      get_type(ret_type, return_type);
      type.return_type() = return_type;

      for (const auto &ptype : func.getParamTypes())
      {
        typet param_type;
        get_type(ptype, param_type);
        type.arguments().push_back(param_type);
      }

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
      get_type(ret_type, return_type);
      type.return_type() = return_type;

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
      get_type(q_typedef_type, new_type);
      break;
    }

    case clang::Type::Record:
    {
      const clang::RecordDecl &tag =
        *(static_cast<const clang::RecordType &>(the_type)).getDecl();

      if(tag.isClass())
      {
        std::cerr << "Class Type is not supported yet" << std::endl;
        abort();
      }

      // Search for the type on the type map
      type_mapt::iterator it;
      search_add_type_map(tag, it);

      symbolt &s = context.symbols.find(it->second)->second;
      new_type = s.type;

      if (tag.isAnonymousStructOrUnion())
        new_type.set("anonymous", true);

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
      get_type(et.getNamedType(), new_type);
      break;
    }

    case clang::Type::TypeOfExpr:
    {
      const clang::TypeOfExprType &tofe =
        static_cast<const clang::TypeOfExprType &>(the_type);

      if(tofe.isSugared())
      {
        get_type(tofe.desugar(), new_type);
      }
      else
      {
        std::cerr << "ESBMC cannot handle desugared TypeOfExpr, "
                  << "please report this benchmark to the developers"
                  << std::endl;
        tofe.dump();
        abort();
      }

      break;
    }

    case clang::Type::TypeOf:
    {
      const clang::TypeOfType &toft =
        static_cast<const clang::TypeOfType &>(the_type);

      if(toft.isSugared())
      {
        get_type(toft.desugar(), new_type);
      }
      else
      {
        std::cerr << "ESBMC cannot handle desugared TypeOf, "
                  << "please report this benchmark to the developers"
                  << std::endl;
        toft.dump();
        abort();
      }

      break;
    }

    case clang::Type::LValueReference:
    {
      const clang::LValueReferenceType &lvrt =
        static_cast<const clang::LValueReferenceType &>(the_type);

      get_type(lvrt.getPointeeTypeAsWritten(), new_type);
      break;
    }

    default:
      std::cerr << "No clang <=> ESBMC migration for type "
                << the_type.getTypeClassName() << std::endl;
      the_type.dump();
      abort();
  }

  if(q_type.isConstQualified())
    new_type.cmt_constant(true);
}

void llvm_convertert::get_builtin_type(
  const clang::BuiltinType& bt,
  typet& new_type)
{
  std::string c_type;

  switch (bt.getKind()) {
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
      c_type = "unsigned char";
      break;

    case clang::BuiltinType::Char16:
      new_type = char16_type();
      c_type = "char16_t";
      break;

    case clang::BuiltinType::Char32:
      new_type = char32_type();
      c_type = "char32_t";
      break;

    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::SChar:
      new_type = signed_char_type();
      c_type = "signed char";
      break;

    case clang::BuiltinType::UShort:
      new_type = unsigned_short_int_type();
      c_type = "unsigned short";
      break;

    case clang::BuiltinType::UInt:
      new_type = uint_type();
      c_type = "unsigned int";
      break;

    case clang::BuiltinType::ULong:
      new_type = long_uint_type();
      c_type = "unsigned long";
      break;

    case clang::BuiltinType::ULongLong:
      new_type = long_long_uint_type();
      c_type = "unsigned long long";
      break;

    case clang::BuiltinType::UInt128:
      // Various simplification / big-int related things use uint64_t's...
      std::cerr << "ESBMC currently does not support integers bigger "
                    "than 64 bits" << std::endl;
      abort();
      break;

    case clang::BuiltinType::Short:
      new_type = signed_short_int_type();
      c_type = "signed short";
      break;

    case clang::BuiltinType::Int:
      new_type = int_type();
      c_type = "signed int";
      break;

    case clang::BuiltinType::Long:
      new_type = long_int_type();
      c_type = "signed long";
      break;

    case clang::BuiltinType::LongLong:
      new_type = long_long_int_type();
      c_type = "signed long long";
      break;

    case clang::BuiltinType::Int128:
      // Various simplification / big-int related things use uint64_t's...
      std::cerr << "ESBMC currently does not support integers bigger "
                    "than 64 bits" << std::endl;
      abort();
      break;

    case clang::BuiltinType::Float:
      new_type = float_type();
      c_type = "float";
      break;

    case clang::BuiltinType::Double:
      new_type = double_type();
      c_type = "double";
      break;

    case clang::BuiltinType::LongDouble:
      new_type = long_double_type();
      c_type = "long double";
      break;

    default:
      std::cerr << "Unrecognized clang builtin type "
      << bt.getName(clang::PrintingPolicy(clang::LangOptions())).str()
      << std::endl;
      abort();
  }

  new_type.set("#c_type", c_type);
}

void llvm_convertert::get_expr(
  const clang::Stmt& stmt,
  exprt& new_expr)
{
  std::string function_name = "";

  const clang::FunctionDecl* fd = get_top_FunctionDecl_from_Stmt(stmt);
  if(fd)
    function_name = fd->getName().str();

  locationt location_begin;
  get_location(
    stmt.getSourceRange().getBegin(),
    function_name,
    location_begin);

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
      get_expr(*opaque_expr.getSourceExpr(), new_expr);
      break;
    }

    // Reference to a declared object, such as functions or variables
    case clang::Stmt::DeclRefExprClass:
    {
      const clang::DeclRefExpr &decl =
        static_cast<const clang::DeclRefExpr&>(stmt);

      const clang::Decl &dcl =
        static_cast<const clang::Decl&>(*decl.getDecl());

      get_decl_ref(dcl, new_expr);
      break;
    }

    // Predefined MACROS as __func__ or __PRETTY_FUNCTION__
    case clang::Stmt::PredefinedExprClass:
    {
      const clang::PredefinedExpr &pred_expr =
        static_cast<const clang::PredefinedExpr&>(stmt);

      get_predefined_expr(pred_expr, new_expr);
      break;
    }

    // An integer value
    case clang::Stmt::IntegerLiteralClass:
    {
      const clang::IntegerLiteral &integer_literal =
        static_cast<const clang::IntegerLiteral&>(stmt);
      llvm::APInt val = integer_literal.getValue();

      if(val.getBitWidth() > 64)
      {
        std::cerr << "ESBMC currently does not support integers bigger "
                     "than 64 bits" << std::endl;
        abort();
      }

      typet the_type;
      get_type(integer_literal.getType(), the_type);

      convert_integer_literal(val, the_type, new_expr);
      break;
    }

    // A character such 'a'
    case clang::Stmt::CharacterLiteralClass:
    {
      const clang::CharacterLiteral &char_literal =
        static_cast<const clang::CharacterLiteral&>(stmt);

      convert_character_literal(char_literal, new_expr);
      break;
    }

    // A float value
    case clang::Stmt::FloatingLiteralClass:
    {
      const clang::FloatingLiteral &float_literal =
        static_cast<const clang::FloatingLiteral&>(stmt);

      typet t;
      get_type(float_literal.getType(), t);

      convert_float_literal(float_literal.getValue(), t, new_expr);
      break;
    }

    // A string
    case clang::Stmt::StringLiteralClass:
    {
      const clang::StringLiteral &string_literal =
        static_cast<const clang::StringLiteral&>(stmt);

      convert_string_literal(string_literal, new_expr);
      break;
    }

    // This is an expr surrounded by parenthesis, we'll ignore it for
    // now, and check its subexpression
    case clang::Stmt::ParenExprClass:
    {
      const clang::ParenExpr& p =
        static_cast<const clang::ParenExpr &>(stmt);
      get_expr(*p.getSubExpr(), new_expr);
      break;
    }

    // An unary operator such as +a, -a, *a or &a
    case clang::Stmt::UnaryOperatorClass:
    {
      const clang::UnaryOperator &uniop =
        static_cast<const clang::UnaryOperator &>(stmt);
      get_unary_operator_expr(uniop, new_expr);
      break;
    }

    // An array subscript operation, such as a[1]
    case clang::Stmt::ArraySubscriptExprClass:
    {
      const clang::ArraySubscriptExpr &arr =
        static_cast<const clang::ArraySubscriptExpr &>(stmt);

      typet t;
      get_type(arr.getType(), t);

      exprt array;
      get_expr(*arr.getLHS(), array);

      exprt pos;
      get_expr(*arr.getRHS(), pos);

      new_expr = index_exprt(array, pos, t);
      break;
    }

    // Support for __builtin_offsetof();
    case clang::Stmt::OffsetOfExprClass:
    {
      const clang::OffsetOfExpr &offset =
        static_cast<const clang::OffsetOfExpr &>(stmt);

      // Use LLVM to calculate offsetof
      llvm::APSInt val;
      assert(offset.EvaluateAsInt(val, *ASTContext));

      new_expr =
        constant_exprt(
          integer2binary(val.getSExtValue(), bv_width(int_type())),
          integer2string(val.getSExtValue()),
          int_type());
      break;
    }

    case clang::Stmt::UnaryExprOrTypeTraitExprClass:
    {
      const clang::UnaryExprOrTypeTraitExpr &unary =
        static_cast<const clang::UnaryExprOrTypeTraitExpr &>(stmt);

      // Use LLVM to calculate sizeof/alignof
      llvm::APSInt val;
      if(unary.EvaluateAsInt(val, *ASTContext))
      {
        new_expr =
          constant_exprt(
            integer2binary(val.getZExtValue(), bv_width(uint_type())),
            integer2string(val.getZExtValue()),
            uint_type());
      }
      else
      {
        assert(unary.getKind() == clang::UETT_SizeOf);

        typet t;
        get_type(unary.getType(), t);

        new_expr = exprt("sizeof", t);
      }

      typet size_type;
      get_type(unary.getTypeOfArgument(), size_type);

      if(size_type.is_struct() || size_type.is_union())
      {
        struct_union_typet t = to_struct_union_type(size_type);
        size_type = symbol_typet("c::" + t.tag().as_string());
      }

      new_expr.set("#c_sizeof_type", size_type);
      break;
    }

    // A function call expr. The symbol may be undefined so we create it here
    // This should be moved to a step after the conversion. The conversion
    // step should only convert the code
    case clang::Stmt::CallExprClass:
    {
      const clang::CallExpr &function_call =
        static_cast<const clang::CallExpr &>(stmt);

      const clang::Stmt *callee = function_call.getCallee();
      exprt callee_expr;
      get_expr(*callee, callee_expr);

      typet type;
      get_type(function_call.getType(), type);

      side_effect_expr_function_callt call;
      call.function() = callee_expr;
      call.type() = type;

      for (const clang::Expr *arg : function_call.arguments()) {
        exprt single_arg;
        get_expr(*arg, single_arg);
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
      get_expr(*member.getBase(), base);

      // If this is anonymous, than get the name from the tag
      if(base.type().get_bool("anonymous"))
        base.component_name(base.type().tag());

      exprt comp;
      get_decl(*member.getMemberDecl(), comp);

      new_expr = member_exprt(base, comp.name(), comp.type());
      break;
    }

    case clang::Stmt::CompoundLiteralExprClass:
    {
      const clang::CompoundLiteralExpr &compound =
        static_cast<const clang::CompoundLiteralExpr &>(stmt);

      exprt initializer;
      get_expr(*compound.getInitializer(), initializer);

      new_expr = initializer;
      break;
    }

    case clang::Stmt::AddrLabelExprClass:
    {
      std::cerr << "ESBMC currently does not support label as values"
                << std::endl;
      abort();

      const clang::AddrLabelExpr &addrlabelExpr =
        static_cast<const clang::AddrLabelExpr &>(stmt);

      exprt label;
      get_decl(*addrlabelExpr.getLabel(), label);

      new_expr = address_of_exprt(label);
      break;
    }

    case clang::Stmt::StmtExprClass:
    {
      const clang::StmtExpr &stmtExpr =
        static_cast<const clang::StmtExpr &>(stmt);

      typet t;
      get_type(stmtExpr.getType(), t);

      exprt subStmt;
      get_expr(*stmtExpr.getSubStmt(), subStmt);

      side_effect_exprt stmt_expr("statement_expression", t);
      stmt_expr.copy_to_operands(subStmt);

      new_expr = stmt_expr;
      break;
    }

    // Casts expression:
    // Implicit: float f = 1; equivalent to float f = (float) 1;
    // CStyle: int a = (int) 3.0;
    case clang::Stmt::ImplicitCastExprClass:
    case clang::Stmt::CStyleCastExprClass:
    {
      const clang::CastExpr &cast =
        static_cast<const clang::CastExpr &>(stmt);
      get_cast_expr(cast, new_expr);
      break;
    }

    // Binary expression such as a+1, a-1 and assignments
    case clang::Stmt::BinaryOperatorClass:
    case clang::Stmt::CompoundAssignOperatorClass:
    {
      const clang::BinaryOperator &binop =
        static_cast<const clang::BinaryOperator&>(stmt);
      get_binary_operator_expr(binop, new_expr);
      break;
    }

    // This is the ternary if
    case clang::Stmt::ConditionalOperatorClass:
    {
      const clang::ConditionalOperator &ternary_if =
        static_cast<const clang::ConditionalOperator &>(stmt);

      exprt cond;
      get_expr(*ternary_if.getCond(), cond);

      exprt then;
      get_expr(*ternary_if.getTrueExpr(), then);

      exprt else_expr;
      get_expr(*ternary_if.getFalseExpr(), else_expr);

      typet t;
      get_type(ternary_if.getType(), t);

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
      get_expr(*ternary_if.getCond(), cond);

      exprt else_expr;
      get_expr(*ternary_if.getFalseExpr(), else_expr);

      typet t;
      get_type(ternary_if.getType(), t);

      side_effect_exprt gcc_ternary("gcc_conditional_expression");
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
      get_type(init_stmt.getType(), t);

      exprt inits;

      // Structs/unions/arrays put the initializer on operands
      if(t.is_struct() || t.is_union() || t.is_array())
      {
        inits = gen_zero(t);

        unsigned int num = init_stmt.getNumInits();
        for (unsigned int i = 0; i < num; i++)
        {
          exprt init;
          get_expr(*init_stmt.getInit(i), init);

          typet elem_type;

          if(t.is_struct() || t.is_union())
            elem_type = to_struct_union_type(t).components()[i].type();
          else
            elem_type = to_array_type(t).subtype();

          gen_typecast(ns, init, elem_type);

          inits.operands().at(i) = init;
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
        get_expr(*init_stmt.getInit(0), inits);
      }

      new_expr = inits;
      break;
    }

    case clang::Stmt::ImplicitValueInitExprClass:
    {
      const clang::ImplicitValueInitExpr &init_stmt =
        static_cast<const clang::ImplicitValueInitExpr &>(stmt);

      typet t;
      get_type(init_stmt.getType(), t);

      new_expr = gen_zero(t);
      break;
    }

    case clang::Stmt::GenericSelectionExprClass:
    {
      const clang::GenericSelectionExpr &gen =
        static_cast<const clang::GenericSelectionExpr&>(stmt);
      get_expr(*gen.getResultExpr(), new_expr);
      break;
    }

    case clang::Stmt::VAArgExprClass:
    {
      const clang::VAArgExpr &vaa =
        static_cast<const clang::VAArgExpr&>(stmt);

      exprt expr;
      get_expr(*vaa.getSubExpr(), expr);

      typet t;
      get_type(vaa.getType(), t);

      exprt vaa_expr("builtin_va_arg", t);
      vaa_expr.copy_to_operands(expr);

      new_expr = vaa_expr;
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
      const clang::DeclStmt &decl =
        static_cast<const clang::DeclStmt&>(stmt);

      const auto &declgroup = decl.getDeclGroup();

      codet decls("decl-block");
      for (clang::DeclGroupRef::const_iterator
        it = declgroup.begin();
        it != declgroup.end();
        it++)
      {
        exprt single_decl;
        get_decl(**it, single_decl);
        decls.operands().push_back(single_decl);
      }

      new_expr = decls;
      break;
    }

    // A NULL statement, we ignore it. An example is a lost semicolon on
    // the program
    case clang::Stmt::NullStmtClass:
      new_expr = code_skipt();
      break;

    // A compound statement is a scope/block
    case clang::Stmt::CompoundStmtClass:
    {
      const clang::CompoundStmt &compound_stmt =
        static_cast<const clang::CompoundStmt &>(stmt);

      code_blockt block;
      for (const auto &stmt : compound_stmt.body()) {
        exprt statement;
        get_expr(*stmt, statement);
        convert_expression_to_code(statement);

        block.operands().push_back(statement);
      }

      // Set the end location for blocks
      locationt location_end;
      get_location(
        stmt.getSourceRange().getEnd(),
        function_name,
        location_end);
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
      get_expr(*case_stmt.getLHS(), value);

      exprt sub_stmt;
      get_expr(*case_stmt.getSubStmt(), sub_stmt);
      convert_expression_to_code(sub_stmt);

      codet label("label");
      exprt &case_ops=label.add_expr("case");
      case_ops.copy_to_operands(value);

      label.copy_to_operands(sub_stmt);

      new_expr = label;
      break;
    }

    // A default statement inside a switch. Same as before, we construct
    // as a label, the difference is that we set default to true
    case clang::Stmt::DefaultStmtClass:
    {
      const clang::DefaultStmt &default_stmt =
        static_cast<const clang::DefaultStmt &>(stmt);

      exprt sub_stmt;
      get_expr(*default_stmt.getSubStmt(), sub_stmt);
      convert_expression_to_code(sub_stmt);

      codet label("label");
      label.set("default", true);
      label.copy_to_operands(sub_stmt);

      new_expr = label;
      break;
    }

    // A label on the program
    case clang::Stmt::LabelStmtClass:
    {
      const clang::LabelStmt &label_stmt =
        static_cast<const clang::LabelStmt &>(stmt);

      exprt sub_stmt;
      get_expr(*label_stmt.getSubStmt(), sub_stmt);
      convert_expression_to_code(sub_stmt);

      codet label("label");
      label.set("label", label_stmt.getName());
      label.copy_to_operands(sub_stmt);

      new_expr = label;
      break;
    }

    // An if then else statement. The else statement may not
    // exist, so we must check before constructing its exprt.
    // We always to try to cast its condition to bool
    case clang::Stmt::IfStmtClass:
    {
      const clang::IfStmt &ifstmt =
        static_cast<const clang::IfStmt &>(stmt);

      exprt cond;
      get_expr(*ifstmt.getCond(), cond);

      exprt then;
      get_expr(*ifstmt.getThen(), then);
      convert_expression_to_code(then);

      codet if_expr("ifthenelse");
      if_expr.copy_to_operands(cond, then);

      if(ifstmt.getElse())
      {
        exprt else_expr;
        get_expr(*ifstmt.getElse(), else_expr);
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

      exprt value;
      get_expr(*switch_stmt.getCond(), value);

      codet body;
      get_expr(*switch_stmt.getBody(), body);

      code_switcht switch_code;
      switch_code.value() = value;
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

      exprt cond;
      get_expr(*while_stmt.getCond(), cond);

      codet body = code_skipt();
      get_expr(*while_stmt.getBody(), body);
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
      const clang::DoStmt &do_stmt =
        static_cast<const clang::DoStmt &>(stmt);

      exprt cond;
      get_expr(*do_stmt.getCond(), cond);

      codet body = code_skipt();
      get_expr(*do_stmt.getBody(), body);
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
      const clang::ForStmt &for_stmt =
        static_cast<const clang::ForStmt &>(stmt);

      codet init = code_skipt();
      if(for_stmt.getInit())
        get_expr(*for_stmt.getInit(), init);
      convert_expression_to_code(init);

      exprt cond = true_exprt();
      if(for_stmt.getCond())
        get_expr(*for_stmt.getCond(), cond);

      codet inc = code_skipt();
      if(for_stmt.getInc())
        get_expr(*for_stmt.getInc(), inc);
      convert_expression_to_code(inc);

      codet body = code_skipt();
      get_expr(*for_stmt.getBody(), body);
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
      std::cerr << "ESBMC currently does not support indirect gotos"
                << std::endl;
      abort();

      const clang::IndirectGotoStmt &goto_stmt =
        static_cast<const clang::IndirectGotoStmt &>(stmt);

      // LLVM was able to compute the target, so this became a
      // common goto
      if(goto_stmt.getConstantTarget())
      {
        code_gotot code_goto;
        code_goto.set_destination(goto_stmt.getConstantTarget()->getName().str());

        new_expr = code_goto;
      }
      else
      {
        exprt target;
        get_expr(*goto_stmt.getTarget(), target);

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
      const clang::ReturnStmt &ret =
        static_cast<const clang::ReturnStmt&>(stmt);

      const clang::Decl *decl = get_top_FunctionDecl_from_Stmt(ret);
      if(!decl)
      {
        std::cerr << "ESBMC could not find the parent scope for "
                  << "the following return statement:" << std::endl;
        ret.dump();
        abort();
      }

      const clang::FunctionDecl &fd =
        static_cast<const clang::FunctionDecl&>(*decl);

      typet return_type;
      get_type(fd.getReturnType(), return_type);

      code_returnt ret_expr;
      if(ret.getRetValue())
      {
        const clang::Expr &retval = *ret.getRetValue();

        exprt val;
        get_expr(retval, val);
        gen_typecast(ns, val, return_type);

        ret_expr.return_value() = val;
      }

      new_expr = ret_expr;
      break;
    }

    // GCC or MS Assembly instruction. We ignore them
    case clang::Stmt::GCCAsmStmtClass:
    case clang::Stmt::MSAsmStmtClass:
      new_expr = code_skipt();
      break;

    // No idea when these AST is created
    case clang::Stmt::ImaginaryLiteralClass:
    case clang::Stmt::ShuffleVectorExprClass:
    case clang::Stmt::ConvertVectorExprClass:
    case clang::Stmt::ChooseExprClass:
    case clang::Stmt::GNUNullExprClass:
    case clang::Stmt::DesignatedInitExprClass:
    case clang::Stmt::ParenListExprClass:
    case clang::Stmt::ExtVectorElementExprClass:
    case clang::Stmt::BlockExprClass:
    case clang::Stmt::AsTypeExprClass:
    case clang::Stmt::PseudoObjectExprClass:
    case clang::Stmt::AtomicExprClass:
    case clang::Stmt::AttributedStmtClass:
    default:
      std::cerr << "Conversion of unsupported clang expr: \"";
      std::cerr << stmt.getStmtClassName() << "\" to expression" << std::endl;
      stmt.dumpColor();
      abort();
  }

  new_expr.location() = location_begin;
}

void llvm_convertert::get_decl_ref(
  const clang::Decl& decl,
  exprt& new_expr)
{
  std::string identifier;
  typet type;

  switch(decl.getKind())
  {
    case clang::Decl::Var:
    {
      const clang::VarDecl &vd =
        static_cast<const clang::VarDecl&>(decl);

      std::size_t address = reinterpret_cast<std::size_t>(&vd);
      identifier = object_map.find(address)->second;

      get_type(vd.getType(), type);
      break;
    }

    case clang::Decl::ParmVar:
    {
      const clang::ParmVarDecl &vd =
        static_cast<const clang::ParmVarDecl&>(decl);

      std::size_t address = reinterpret_cast<std::size_t>(&vd);
      identifier = object_map.find(address)->second;

      get_type(vd.getType(), type);
      break;
    }

    case clang::Decl::Function:
    {
      const clang::FunctionDecl &fd =
        static_cast<const clang::FunctionDecl&>(decl);

      std::size_t address = reinterpret_cast<std::size_t>(&fd);

      // We may not find the function's symbol, because it is
      // undefined or not defined at all
      object_mapt::iterator it = object_map.find(address);
      if(it == object_map.end())
      {
        identifier =
          get_default_name(fd.getName().str(), !fd.isExternallyVisible());
      }
      else
      {
        identifier = it->second;
      }

      get_type(fd.getType(), type);
      break;
    }

    case clang::Decl::EnumConstant:
    {
      const clang::EnumConstantDecl &enumcd =
        static_cast<const clang::EnumConstantDecl &>(decl);

      // For enum constants, we get their value directly
      new_expr =
        constant_exprt(
          integer2binary(enumcd.getInitVal().getSExtValue(),
            bv_width(int_type())),
          integer2string(enumcd.getInitVal().getSExtValue()),
          int_type());
      return;

      break;
    }

    default:
      std::cerr << "Conversion of unsupported clang decl ref: \"";
      std::cerr << decl.getDeclKindName() << "\" to expression" << std::endl;
      decl.dump();
      abort();
  }

  new_expr = exprt("symbol", type);
  new_expr.identifier(identifier);
  new_expr.cmt_lvalue(true);

  if(identifier.find_last_of("::") != std::string::npos)
    new_expr.name(identifier.substr(identifier.find_last_of("::")+1));
  else
    new_expr.name(identifier);
}

void llvm_convertert::get_cast_expr(
  const clang::CastExpr& cast,
  exprt& new_expr)
{
  exprt expr;
  get_expr(*cast.getSubExpr(), expr);

  typet type;
  get_type(cast.getType(), type);

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
      abort();
  }

  new_expr = expr;
}

void llvm_convertert::get_unary_operator_expr(
  const clang::UnaryOperator& uniop,
  exprt& new_expr)
{
  typet uniop_type;
  get_type(uniop.getType(), uniop_type);

  exprt unary_sub;
  get_expr(*uniop.getSubExpr(), unary_sub);

  switch (uniop.getOpcode())
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

    default:
      std::cerr << "Conversion of unsupported clang unary operator: \"";
      std::cerr << clang::UnaryOperator::getOpcodeStr(uniop.getOpcode()).str()
                << "\" to expression" << std::endl;
      uniop.dumpColor();
      abort();
  }

  new_expr.operands().push_back(unary_sub);
}

void llvm_convertert::get_binary_operator_expr(
  const clang::BinaryOperator& binop,
  exprt& new_expr)
{
  switch(binop.getOpcode())
  {
    case clang::BO_Add:
      new_expr = exprt("+");
      break;

    case clang::BO_Sub:
      new_expr = exprt("-");
      break;

    case clang::BO_Mul:
      new_expr = exprt("*");
      break;

    case clang::BO_Div:
      new_expr = exprt("/");
      break;

    case clang::BO_Shl:
      new_expr = exprt("shl");
      break;

    case clang::BO_Shr:
      new_expr = exprt("shr");
      break;

    case clang::BO_Rem:
      new_expr = exprt("mod");
      break;

    case clang::BO_And:
      new_expr = exprt("bitand");
      break;

    case clang::BO_Xor:
      new_expr = exprt("bitxor");
      break;

    case clang::BO_Or:
      new_expr = exprt("bitor");
      break;

    case clang::BO_LT:
      new_expr = exprt("<");
      break;

    case clang::BO_GT:
      new_expr = exprt(">");
      break;

    case clang::BO_LE:
      new_expr = exprt("<=");
      break;

    case clang::BO_GE:
      new_expr = exprt(">=");
      break;

    case clang::BO_EQ:
      new_expr = exprt("=");
      break;

    case clang::BO_NE:
      new_expr = exprt("notequal");
      break;

    case clang::BO_LAnd:
      new_expr = exprt("and");
      break;

    case clang::BO_LOr:
      new_expr = exprt("or");
      break;

    case clang::BO_Assign:
      // If we use code_assignt, it will reserve two operands,
      // and the copy_to_operands method call at the end of
      // this method will put lhs and rhs in positions 2 and 3,
      // instead of 0 and 1 :/
      new_expr = side_effect_exprt("assign");
      break;

    case clang::BO_Comma:
      new_expr = exprt("comma");
      break;

    default:
    {
      const clang::CompoundAssignOperator &compop =
        static_cast<const clang::CompoundAssignOperator &>(binop);
      get_compound_assign_expr(compop, new_expr);
      return;
    }
  }

  exprt lhs;
  get_expr(*binop.getLHS(), lhs);

  exprt rhs;
  get_expr(*binop.getRHS(), rhs);

  get_type(binop.getType(), new_expr.type());

  new_expr.copy_to_operands(lhs, rhs);
}

void llvm_convertert::get_compound_assign_expr(
  const clang::CompoundAssignOperator& compop,
  exprt& new_expr)
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
      abort();
  }

  exprt lhs;
  get_expr(*compop.getLHS(), lhs);

  exprt rhs;
  get_expr(*compop.getRHS(), rhs);

  get_type(compop.getType(), new_expr.type());

  if(!lhs.type().is_pointer())
    gen_typecast(ns, rhs, lhs.type());

  new_expr.copy_to_operands(lhs, rhs);
}

void llvm_convertert::get_predefined_expr(
  const clang::PredefinedExpr& pred_expr,
  exprt& new_expr)
{
  typet t;
  get_type(pred_expr.getType(), t);

  switch (pred_expr.getIdentType())
  {
    case clang::PredefinedExpr::Func:
    case clang::PredefinedExpr::Function:
    case clang::PredefinedExpr::LFunction:
    case clang::PredefinedExpr::FuncDName:
    case clang::PredefinedExpr::FuncSig:
    case clang::PredefinedExpr::PrettyFunction:
    case clang::PredefinedExpr::PrettyFunctionNoVirtual:
      break;
    default:
      std::cerr << "Conversion of unsupported clang predefined expr: \""
        << pred_expr.getIdentType() << "\" to expression" << std::endl;
      pred_expr.dumpColor();
      abort();
  }

  string_constantt string;
  string.set_value(pred_expr.getFunctionName()->getString().str());

  new_expr.swap(string);
}

void llvm_convertert::get_default_symbol(
  symbolt& symbol,
  typet type,
  std::string base_name,
  std::string pretty_name,
  locationt location,
  bool is_local)
{
  symbol.mode = "C";
  symbol.module = get_modulename_from_path();
  symbol.location = location;
  symbol.type = type;
  symbol.base_name = base_name;
  symbol.pretty_name = pretty_name;
  symbol.name = get_default_name(pretty_name, is_local);
}

std::string llvm_convertert::get_default_name(
  std::string name,
  bool is_local)
{
  std::string symbol_name = "c::";

  if(is_local)
    symbol_name += get_modulename_from_path() + "::";

  symbol_name += name;

  return symbol_name;
}

std::string llvm_convertert::get_var_name(
  std::string name,
  std::string function_name)
{
  std::string pretty_name = "";

  if(name.empty())
    pretty_name = "#anon"+i2string(anon_counter++);

  // This means a global/static variable
  if(!function_name.empty())
  {
    pretty_name += function_name + "::";
    pretty_name += integer2string(current_scope_var_num++) + "::";
  }

  pretty_name += name;
  return pretty_name;
}

std::string llvm_convertert::get_param_name(
  std::string name,
  std::string function_name)
{
  std::string pretty_name = get_modulename_from_path() + "::";
  pretty_name += function_name + "::";
  pretty_name += name;

  return pretty_name;
}

std::string llvm_convertert::get_tag_name(
  const clang::RecordDecl& recordd)
{
  std::string pretty_name = "tag-";

  if(recordd.getName().str().empty())
  {
    clang::RecordDecl *record_def = recordd.getDefinition();
    struct_typet t;
    get_struct_union_class_fields(*record_def, t);

    pretty_name = "anon#" + type2name(t);
  }
  else
  {
    pretty_name += recordd.getName().str();
  }

  return pretty_name;
}

void llvm_convertert::get_location_from_decl(
  const clang::Decl& decl,
  locationt &location_begin)
{
  sm = &ASTContext->getSourceManager();

  std::string function_name = "";

  if(decl.getDeclContext()->isFunctionOrMethod())
  {
    const clang::FunctionDecl &funcd =
      static_cast<const clang::FunctionDecl &>(*decl.getDeclContext());

    function_name = funcd.getName().str();
  }

  get_location(
    decl.getSourceRange().getBegin(),
    function_name,
    location_begin);
}

void llvm_convertert::get_location(
  clang::SourceLocation loc,
  std::string function_name,
  locationt &location)
{
  if(!sm)
    return;

  clang::SourceLocation SpellingLoc = sm->getSpellingLoc(loc);
  clang::PresumedLoc PLoc = sm->getPresumedLoc(SpellingLoc);

  if (PLoc.isInvalid()) {
    location.set_file("<invalid sloc>");
    return;
  }

  current_path = PLoc.getFilename();

  location.set_line(PLoc.getLine());
  location.set_file(get_filename_from_path());

  if(!function_name.empty())
    location.set_function(function_name);
}

std::string llvm_convertert::get_modulename_from_path()
{
  return boost::filesystem::path(current_path).stem().string();
}

std::string llvm_convertert::get_filename_from_path()
{
  return boost::filesystem::path(current_path).filename().string();
}

void llvm_convertert::move_symbol_to_context(
  symbolt& symbol)
{
  symbolst::iterator old_it=context.symbols.find(symbol.name);
  if(old_it==context.symbols.end())
  {
    if (context.move(symbol))
    {
      std::cerr << "Couldn't add symbol " << symbol.name
          << " to symbol table" << std::endl;
      symbol.dump();
      abort();
    }
  }
  else
  {
    symbolt &old_symbol = old_it->second;
    check_symbol_redefinition(old_symbol, symbol);
  }
}

void llvm_convertert::dump_type_map()
{
  std::cout << "Type_map:" << std::endl;
  for (auto it : type_map)
    std::cout << it.first << ": " << it.second << std::endl;
}

void llvm_convertert::dump_object_map()
{
  std::cout << "Object_map:" << std::endl;
  for (auto it : object_map)
    std::cout << it.first << ": " << it.second << std::endl;
}

void llvm_convertert::check_symbol_redefinition(
  symbolt& old_symbol,
  symbolt& new_symbol)
{
  // types that are code means functions
  if(old_symbol.type.is_code())
  {
    if(new_symbol.value.is_not_nil())
    {
      if(old_symbol.value.is_not_nil())
      {
        // If this is a invalid redefinition, LLVM will complain and
        // won't convert the program. We should safely to ignore this.
      }
      else
      {
        // overwrite
        old_symbol.swap(new_symbol);
      }
    }
  }
  else if(old_symbol.is_type)
  {
    if(new_symbol.type.is_not_nil())
    {
      if(old_symbol.type.is_not_nil())
      {
        // If this is a invalid redefinition, LLVM will complain and
        // won't convert the program. We should safely to ignore this.
      }
      else
      {
        // overwrite
        old_symbol.swap(new_symbol);
      }
    }
  }
}

void llvm_convertert::convert_expression_to_code(exprt& expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

void llvm_convertert::search_add_type_map(
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
    get_decl(tag, decl);
  }

  type_it = type_map.find(address);
  if(type_it == type_map.end())
  {
    // BUG! This should be added already
    abort();
  }
}

const clang::Decl* llvm_convertert::get_DeclContext_from_Stmt(
  const clang::Stmt& stmt)
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

const clang::FunctionDecl* llvm_convertert::get_top_FunctionDecl_from_Stmt(
  const clang::Stmt& stmt)
{
  const clang::Decl *decl = get_DeclContext_from_Stmt(stmt);
  if(decl)
    return static_cast<const clang::FunctionDecl*>(decl->getNonClosureContext());

  return nullptr;
}
