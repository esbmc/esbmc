/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_convert.h"

#include <std_code.h>
#include <expr_util.h>

#include <ansi-c/c_types.h>

#include <boost/filesystem.hpp>

llvm_convertert::llvm_convertert(contextt &_context)
  : context(_context)
{
}

llvm_convertert::~llvm_convertert()
{
}

bool llvm_convertert::convert()
{
  if(convert_top_level_decl())
    return true;

  return false;
}

bool llvm_convertert::convert_top_level_decl()
{
  // Iterate through each translation unit and their global symbols, creating
  // symbols as we go.

  for (auto &translation_unit : ASTs) {

    clang::ASTUnit::top_level_iterator it;
    for (it = translation_unit->top_level_begin();
        it != translation_unit->top_level_end(); it++) {

      switch ((*it)->getKind()) {
        case clang::Decl::Typedef:
          convert_typedef(it);
          break;

        case clang::Decl::Var:
          convert_var(it);
          break;

        case clang::Decl::Function:
          convert_function(it);
          break;

        // Apparently if you insert a semicolon at the end of a
        // function declaration, this AST is created, so just
        // ignore it
        case clang::Decl::Empty:
          break;

        case clang::Decl::Record:
        default:
          std::cerr << "Unrecognized / unimplemented decl type ";
          std::cerr << (*it)->getDeclKindName() << std::endl;
          abort();
      }
    }
  }

  return false;
}

void llvm_convertert::convert_typedef(clang::ASTUnit::top_level_iterator it)
{
  symbolt symbol;
  get_default_symbol(symbol, it);

  clang::TypedefDecl *tdd = dynamic_cast<clang::TypedefDecl*>(*it);
  clang::QualType q_type = tdd->getUnderlyingType();
  const clang::Type *the_type = q_type.getTypePtrOrNull();
  assert(the_type != NULL && "No underlying typedef type?");

  // Get type
  typet t;
  get_type(*the_type, t);

  symbol.type = t;
  symbol.base_name = tdd->getName().str();
  symbol.pretty_name =
      symbol.module.as_string() + "::" + symbol.base_name.as_string();
  symbol.name =
      "c::" + symbol.module.as_string() + "::" + symbol.base_name.as_string();
  symbol.is_type = true;

  if (context.move(symbol)) {
    std::cerr << "Couldn't add symbol " << symbol.name
              << " to symbol table" << std::endl;
    abort();
  }
}

void llvm_convertert::convert_var(clang::ASTUnit::top_level_iterator it)
{
  symbolt symbol;
  get_default_symbol(symbol, it);

  clang::VarDecl *vd = dynamic_cast<clang::VarDecl*>(*it);
  clang::QualType q_type = vd->getType();
  const clang::Type *the_type = q_type.getTypePtrOrNull();
  assert(the_type != NULL && "No type?");

  // Get type
  typet t;
  get_type(*the_type, t);

  symbol.type = t;
  symbol.base_name = vd->getName().str();

  // This is not local, so has static lifetime
  if (!vd->hasLocalStorage())
  {
    symbol.static_lifetime = true;
    symbol.name = "c::" + symbol.base_name.as_string();
    symbol.pretty_name = symbol.base_name.as_string();
    symbol.value = gen_zero(t);
  }
  else
  {
    symbol.name =
      "c::" + symbol.module.as_string() + "::" + symbol.base_name.as_string();
    symbol.pretty_name =
      symbol.module.as_string() + "::" + symbol.base_name.as_string();
  }

  if (vd->hasExternalStorage()) {
    symbol.is_extern = true;
  }

  symbol.lvalue = true;

  if (context.move(symbol)) {
    std::cerr << "Couldn't add symbol " << symbol.name
              << " to symbol table" << std::endl;
    abort();
  }
}

void llvm_convertert::convert_function(clang::ASTUnit::top_level_iterator it)
{
  symbolt symbol;
  get_default_symbol(symbol, it);

  clang::FunctionDecl *fd = dynamic_cast<clang::FunctionDecl*>(*it);

  symbol.base_name = fd->getName().str();
  symbol.name = "c::" + symbol.base_name.as_string();
  symbol.pretty_name = symbol.base_name.as_string();
  symbol.lvalue = true;

  // We need: a type, a name, and an optional body
  clang::Stmt *body = NULL;
  if (fd->isThisDeclarationADefinition() && fd->hasBody())
    body = fd->getBody();

  // Build function's type
  code_typet type;

  // Return type
  const clang::Type *ret_type = fd->getReturnType().getTypePtr();
  typet return_type;
  get_type(*ret_type, return_type);
  type.return_type() = return_type;

  // The arguments
  type.arguments();
  if(body)
  {
    for (const auto &pdecl : fd->params()) {
      code_typet::argumentt param =
        convert_function_params(symbol.base_name.as_string(), it, pdecl);
      type.arguments().push_back(param);
    }
  }

  // And the location
  type.location() = symbol.location;
  symbol.type = type;

  if (context.move(symbol)) {
    std::cerr << "Couldn't add symbol " << symbol.name
              << " to symbol table" << std::endl;
    abort();
  }
}

code_typet::argumentt llvm_convertert::convert_function_params(
  std::string function_name,
  clang::ASTUnit::top_level_iterator it,
  clang::ParmVarDecl *pdecl)
{
  symbolt param_symbol;
  get_default_symbol(param_symbol, it);

  const clang::Type *par_type = pdecl->getOriginalType().getTypePtr();
  typet param_type;
  get_type(*par_type, param_type);

  param_symbol.type = param_type;

  std::string name = pdecl->getNameAsString();
  param_symbol.pretty_name = function_name + "::" + name;
  param_symbol.name = "c::" + param_symbol.pretty_name.as_string();
  param_symbol.base_name = name;

  param_symbol.lvalue = true;
  param_symbol.file_local = true;
  param_symbol.is_actual = true;

  if (context.move(param_symbol)) {
    std::cerr << "Couldn't add symbol " << param_symbol.name
        << " to symbol table" << std::endl;
    abort();
  }

  code_typet::argumentt arg;
  arg.type() = param_type;
  arg.base_name(name);
  arg.identifier(param_symbol.name.as_string());
  arg.location() = param_symbol.location;

  return arg;
}

void llvm_convertert::get_type(const clang::Type &the_type, typet &new_type)
{
  clang::Type::TypeClass tc = the_type.getTypeClass();
  switch (tc) {
    case clang::Type::Builtin:
    {
      const clang::BuiltinType &bt = static_cast<const clang::BuiltinType&>(the_type);
      switch (bt.getKind()) {
        case clang::BuiltinType::Void:
          new_type = empty_typet();
          break;

        case clang::BuiltinType::Bool:
          new_type = bool_type();
          break;

        case clang::BuiltinType::UChar:
          new_type = unsignedbv_typet(config.ansi_c.char_width);
          break;

        case clang::BuiltinType::Char16:
          new_type = unsignedbv_typet(16);
          break;

        case clang::BuiltinType::Char32:
          new_type = unsignedbv_typet(32);
          break;

        case clang::BuiltinType::UShort:
          new_type = unsignedbv_typet(config.ansi_c.short_int_width);
          break;

        case clang::BuiltinType::UInt:
          new_type = uint_type();
          break;

        case clang::BuiltinType::ULong:
          new_type = long_uint_type();
          break;

        case clang::BuiltinType::ULongLong:
          new_type = long_long_uint_type();
          break;

        case clang::BuiltinType::UInt128:
          // Various simplification / big-int related things use uint64_t's...
          std::cerr << "No support for uint128's in ESBMC right now, sorry" << std::endl;
          abort();

        case clang::BuiltinType::SChar:
          new_type = signedbv_typet(config.ansi_c.char_width);
          break;

        case clang::BuiltinType::Short:
          new_type = signedbv_typet(config.ansi_c.short_int_width);
          break;

        case clang::BuiltinType::Int:
          new_type = int_type();
          break;

        case clang::BuiltinType::Long:
          new_type = long_int_type();
          break;

        case clang::BuiltinType::LongLong:
          new_type = long_long_int_type();
          break;

        case clang::BuiltinType::Int128:
          // Various simplification / big-int related things use uint64_t's...
          std::cerr << "No support for uint128's in ESBMC right now, sorry" << std::endl;
          abort();

        case clang::BuiltinType::Float:
          new_type = float_type();
          break;

        case clang::BuiltinType::Double:
          new_type = double_type();
          break;

        case clang::BuiltinType::LongDouble:
          new_type = long_double_type();
          break;

        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::WChar_S:
        case clang::BuiltinType::WChar_U:
        case clang::BuiltinType::NullPtr:
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCClass:
        case clang::BuiltinType::ObjCSel:
        case clang::BuiltinType::OCLImage1d:
        case clang::BuiltinType::OCLImage1dArray:
        case clang::BuiltinType::OCLImage1dBuffer:
        case clang::BuiltinType::OCLImage2d:
        case clang::BuiltinType::OCLImage2dArray:
        case clang::BuiltinType::OCLImage3d:
        case clang::BuiltinType::OCLSampler:
        case clang::BuiltinType::OCLEvent:
        case clang::BuiltinType::Dependent:
        case clang::BuiltinType::Overload:
        case clang::BuiltinType::BoundMember:
        case clang::BuiltinType::PseudoObject:
        case clang::BuiltinType::UnknownAny:
        case clang::BuiltinType::BuiltinFn:
        case clang::BuiltinType::ARCUnbridgedCast:
        case clang::BuiltinType::Half:
          std::cerr << "Unrecognized clang builtin type "
                    << bt.getName(clang::PrintingPolicy(clang::LangOptions())).str()
                    << std::endl;
          abort();
      }
    }
    return;

    case clang::Type::Record:
    case clang::Type::ConstantArray:
    case clang::Type::Elaborated:
    case clang::Type::Pointer:
    case clang::Type::Typedef:
    case clang::Type::FunctionProto:
    case clang::Type::FunctionNoProto:
    case clang::Type::IncompleteArray:
    case clang::Type::Paren:
    default:
      std::cerr << "No clang <=> ESBMC migration for type "
                << the_type.getTypeClassName() << std::endl;
      abort();
  }
}

void llvm_convertert::get_default_symbol(symbolt& symbol, clang::ASTUnit::top_level_iterator it)
{
  std::string path =
    (*it)->getASTContext().getSourceManager().getFilename(
      (*it)->getLocation()).str();

  symbol.mode = "C";
  symbol.module = get_modulename_from_path(path);

  locationt location;
  location.set_file(get_filename_from_path(path));
  location.set_line((*it)->getASTContext().getSourceManager().
    getSpellingLineNumber((*it)->getLocation()));
  symbol.location = location;
}

std::string llvm_convertert::get_filename_from_path(std::string path)
{
  return  boost::filesystem::path(path).filename().string();
}

std::string llvm_convertert::get_modulename_from_path(std::string path)
{
  return  boost::filesystem::path(path).stem().string();
}
