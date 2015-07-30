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

#include <ansi-c/c_types.h>
#include <ansi-c/convert_integer_literal.h>

#include <boost/filesystem.hpp>

std::string repeat( const std::string &word, int times ) {
   std::string result ;
   result.reserve(times*word.length()); // avoid repeated reallocation
   for (int a = 0 ; a < times ; a++)
      result += word ;
   return result ;
}

llvm_convertert::llvm_convertert(contextt &_context)
  : context(_context),
    ns(context),
    current_location(locationt()),
    current_path(""),
    current_function_name(""),
    current_scope(0)
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
  for (auto &translation_unit : ASTs)
  {
    for (clang::ASTUnit::top_level_iterator
      it = translation_unit->top_level_begin();
      it != translation_unit->top_level_end();
      it++)
    {
      update_current_location(it);

      exprt dummy_decl;
      convert_decl(**it, dummy_decl);
    }
  }

  return false;
}

void llvm_convertert::convert_decl(
  const clang::Decl& decl,
  exprt &new_expr)
{
  switch (decl.getKind()) {
    case clang::Decl::Typedef:
    {
      const clang::TypedefDecl &tdd =
        static_cast<const clang::TypedefDecl&>(decl);
      convert_typedef(tdd, new_expr);
      break;
    }

    case clang::Decl::Var:
    {
      const clang::VarDecl &vd =
        static_cast<const clang::VarDecl&>(decl);
      convert_var(vd, new_expr);
      break;
    }

    case clang::Decl::Function:
    {
      const clang::FunctionDecl &fd =
        static_cast<const clang::FunctionDecl&>(decl);
      convert_function(fd);
      break;
    }

    // Apparently if you insert a semicolon at the end of a
    // function declaration, this AST is created, so just
    // ignore it
    case clang::Decl::Empty:
      break;

    case clang::Decl::Record:
    default:
      std::cerr << "Unrecognized / unimplemented decl type ";
      std::cerr << decl.getDeclKindName() << std::endl;
      abort();
  }
}

void llvm_convertert::convert_typedef(
  const clang::TypedefDecl &tdd,
  exprt &new_expr)
{
  symbolt symbol;
  get_default_symbol(symbol);

  clang::QualType q_type = tdd.getUnderlyingType();

  // Get type
  typet t;
  get_type(q_type, t);

  symbol.type = t;
  symbol.base_name = tdd.getName().str();
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

  if(current_function_name!= "")
    new_expr = code_skipt();
}

void llvm_convertert::convert_var(
  const clang::VarDecl &vd,
  exprt &new_expr)
{
  symbolt symbol;
  get_default_symbol(symbol);

  clang::QualType q_type = vd.getType();

  // Get type
  typet t;
  get_type(q_type, t);

  symbol.type = t;
  symbol.base_name = vd.getName().str();

  // This is not local, so has static lifetime
  if (!vd.hasLocalStorage())
  {
    symbol.static_lifetime = true;
    symbol.value = gen_zero(t);

    // Add location to value since it is only added on get_expr
    symbol.value.location() = current_location;
  }

  symbol.pretty_name =
    get_var_name(symbol.base_name.as_string(), vd.hasLocalStorage());
  symbol.name = "c::" + symbol.pretty_name.as_string();

  if(vd.hasInit())
  {
    const clang::Expr *value = vd.getInit();
    get_expr(*value, symbol.value);
  }

  if (vd.hasExternalStorage()) {
    symbol.is_extern = true;
  }

  symbol.lvalue = true;

  code_declt decl;
  decl.operands().push_back(symbol_expr(symbol));
  new_expr = decl;

  if (context.move(symbol)) {
    std::cerr << "Couldn't add symbol " << symbol.name
              << " to symbol table" << std::endl;
    abort();
  }
}

void llvm_convertert::convert_function(const clang::FunctionDecl &fd)
{
  std::string old_function_name = current_function_name;

  symbolt symbol;
  get_default_symbol(symbol);

  symbol.base_name = fd.getName().str();
  symbol.name = "c::" + symbol.base_name.as_string();
  symbol.pretty_name = symbol.base_name.as_string();
  symbol.lvalue = true;

  current_function_name = fd.getName().str();

  // We need: a type, a name, and an optional body
  clang::Stmt *body = NULL;
  if (fd.isThisDeclarationADefinition() && fd.hasBody())
  {
    body = fd.getBody();
    get_expr(*body, symbol.value);
  }

  // Build function's type
  code_typet type;

  // Return type
  const clang::QualType ret_type = fd.getReturnType();
  typet return_type;
  get_type(ret_type, return_type);
  type.return_type() = return_type;

  // The arguments
  if(body)
  {
    for (const auto &pdecl : fd.params()) {
      code_typet::argumentt param =
        convert_function_params(symbol.base_name.as_string(), pdecl);
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

  current_function_name = old_function_name;
}

code_typet::argumentt llvm_convertert::convert_function_params(
  std::string function_name,
  clang::ParmVarDecl *pdecl)
{
  symbolt param_symbol;
  get_default_symbol(param_symbol);

  const clang::QualType q_type = pdecl->getOriginalType();
  typet param_type;
  get_type(q_type, param_type);

  param_symbol.type = param_type;

  std::string name = pdecl->getNameAsString();
  param_symbol.pretty_name = function_name + "::" + name;
  param_symbol.name = "c::" + param_symbol.pretty_name.as_string();
  param_symbol.base_name = name;

  param_symbol.lvalue = true;
  param_symbol.file_local = true;
  param_symbol.is_actual = true;

  code_typet::argumentt arg;
  arg.type() = param_type;
  arg.base_name(name);
  arg.cmt_identifier(param_symbol.name.as_string());
  arg.location() = param_symbol.location;

  if (context.move(param_symbol)) {
    std::cerr << "Couldn't add symbol " << param_symbol.name
        << " to symbol table" << std::endl;
    abort();
  }

  return arg;
}

void llvm_convertert::get_type(const clang::QualType &q_type, typet &new_type)
{
  const clang::Type &the_type = *q_type.getTypePtrOrNull();
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

        default:
          std::cerr << "Unrecognized clang builtin type "
                    << bt.getName(clang::PrintingPolicy(clang::LangOptions())).str()
                    << std::endl;
          abort();
      }
      break;
    }

    case clang::Type::Typedef:
    {
      const clang::TypedefType &pt =
        static_cast<const clang::TypedefType &>(the_type);
      clang::QualType q_typedef_type = pt.getDecl()->getUnderlyingType();
      get_type(q_typedef_type, new_type);
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

void llvm_convertert::get_expr(
  const clang::Stmt& stmt,
  exprt& new_expr)
{
  switch(stmt.getStmtClass()) {
    case clang::Stmt::IntegerLiteralClass:
    {
      const clang::IntegerLiteral &integer =
        static_cast<const clang::IntegerLiteral&>(stmt);
      llvm::APInt val = integer.getValue();
      assert(val.getBitWidth() <= 64 && "Too large an integer found, sorry");

      typet the_type;
      get_type(integer.getType(), the_type);
      assert(the_type.is_unsignedbv() || the_type.is_signedbv());

      exprt bval;
      if (the_type.is_unsignedbv()) {
        uint64_t the_val = val.getZExtValue();
        convert_integer_literal(integer2string(the_val) + "u", bval, 10);
      } else {
        int64_t the_val = val.getSExtValue();
        convert_integer_literal(integer2string(the_val), bval, 10);
      }

      new_expr.swap(bval);
      break;
    }

    case clang::Stmt::ImplicitCastExprClass:
    case clang::Stmt::CStyleCastExprClass:
    {
      const clang::CastExpr &cast = static_cast<const clang::CastExpr &>(stmt);

      typet type;
      get_type(cast.getType(), type);

      exprt expr;
      get_expr(*cast.getSubExpr(), expr);

      new_expr = typecast_exprt(expr, type);
      break;
    }

    case clang::Stmt::CompoundStmtClass:
    {
      const clang::CompoundStmt &compound_stmt =
        static_cast<const clang::CompoundStmt &>(stmt);

      // Increase current scope number, it will be used for variables' name
      // This will be increased every time a block is parsed
      assert(current_scope >= 0);
      ++current_scope;

      code_blockt block;
      for (const auto &stmt : compound_stmt.body()) {
        exprt statement;
        get_expr(*stmt, statement);
        block.operands().push_back(statement);
      }
      new_expr = block;

      --current_scope;
      assert(current_scope >= 0);
      break;
    }

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
        convert_decl(**it, single_decl);
        decls.operands().push_back(single_decl);
      }

      new_expr = decls;
      break;
    }

    case clang::Stmt::BinaryOperatorClass:
    {
      const clang::BinaryOperator &binop =
        static_cast<const clang::BinaryOperator&>(stmt);
      get_binary_operator_expr(binop, new_expr);
      break;
    }

    case clang::Stmt::DeclRefExprClass:
    {
      const clang::DeclRefExpr &decl =
        static_cast<const clang::DeclRefExpr&>(stmt);
      const clang::VarDecl &vd =
        static_cast<const clang::VarDecl&>(*decl.getDecl());

      irep_idt identifier =
        "c::" + get_var_name(vd.getName().str(), vd.hasLocalStorage());

      const symbolt &sym = ns.lookup(identifier);
      new_expr = symbol_expr(sym);
      break;
    }

    default:
      std::cerr << "Conversion of unsupported clang expr: \"";
      std::cerr << stmt.getStmtClassName() << "\" to expression" << std::endl;
      stmt.dump();
      abort();
  }

  new_expr.location() = current_location;
}


void llvm_convertert::get_binary_operator_expr(
  const clang::BinaryOperator& binop,
  exprt& new_expr)
{
  switch(binop.getOpcode())
  {
    default:
      std::cerr << "Conversion of unsupported clang binary operator: \"";
      std::cerr << binop.getOpcodeStr().str() << "\" to expression" << std::endl;
      binop.dump();
      abort();
  }
}

void llvm_convertert::get_default_symbol(symbolt& symbol)
{
  symbol.mode = "C";
  symbol.module = get_modulename_from_path();
  symbol.location = current_location;
}

std::string llvm_convertert::get_var_name(
  std::string name,
  bool is_local)
{
  if(!is_local)
    return name;

  std::string pretty_name = get_modulename_from_path() + "::";
  if(current_function_name!= "")
    pretty_name += current_function_name + "::";
  if(current_scope > 0)
    pretty_name += repeat("1::", current_scope);
  pretty_name += name;

  return pretty_name;
}

void llvm_convertert::update_current_location(
    clang::ASTUnit::top_level_iterator it)
{
  current_path =
    (*it)->getASTContext().getSourceManager().getFilename(
      (*it)->getLocation()).str();

  current_location.set_file(get_filename_from_path());
  current_location.set_line((*it)->getASTContext().getSourceManager().
    getSpellingLineNumber((*it)->getLocation()));
}

std::string llvm_convertert::get_modulename_from_path()
{
  return  boost::filesystem::path(current_path).stem().string();
}

std::string llvm_convertert::get_filename_from_path()
{
  return  boost::filesystem::path(current_path).filename().string();
}
