#ifndef CLANG_C_FRONTEND_CLANG_C_CONVERT_H_
#define CLANG_C_FRONTEND_CLANG_C_CONVERT_H_

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/std_types.h>
#include <util/symbol_generator.h>

// Forward dec, to avoid bringing in clang headers
namespace clang
{
class ASTUnit;
class ASTContext;
class SourceManager;
class FunctionDecl;
class Decl;
class VarDecl;
class ParmVarDecl;
class RecordDecl;
class QualType;
class Type;
class BuiltinType;
class Stmt;
class BinaryOperator;
class CompoundAssignOperator;
class UnaryOperator;
class AtomicExpr;
class CastExpr;
class NamedDecl;
class PresumedLoc;
class SourceLocation;
class CharacterLiteral;
class StringLiteral;
class IntegerLiteral;
class FloatingLiteral;
class TagDecl;
} // namespace clang

class clang_c_convertert
{
public:
  clang_c_convertert(
    contextt &_context,
    std::vector<std::unique_ptr<clang::ASTUnit>> &_ASTs,
    irep_idt _mode);
  virtual ~clang_c_convertert() = default;

  bool convert();

  /**
 * @brief Perform the typecast by creating a tmp variable on RHS
 *
 * The idea is to look for all components of the union and match
 * the type. If not found, throws an error
 *
 * @param dest RHS dest
 * @param type Union type
 * @param msg  Message object
 */
  static void gen_typecast_to_union(exprt &dest, const typet &type);

  static std::string get_decl_name(const clang::NamedDecl &nd);

protected:
  clang::ASTContext *ASTContext;
  contextt &context;
  namespacet ns;
  std::vector<std::unique_ptr<clang::ASTUnit>> &ASTs;
  irep_idt mode;
  symbol_generator anon_symbol;

  unsigned int current_scope_var_num;
  /**
   *  During get_expr(), which also transforms blocks/scopes, this represents
   *  the latest opened blocks from the top-level. A nullptr 'current_block'
   *  thus means file scope.
   */
  code_blockt *current_block;

  clang::SourceManager *sm;

  const clang::FunctionDecl *current_functionDecl;

  bool convert_builtin_types();
  bool convert_top_level_decl();

  /**
   *  Since this class is inherited by clang-cpp-frontend,
   *  some get_* functions are made `virtual' to deal with clang CXX declarations
   */
  virtual bool get_decl(const clang::Decl &decl, exprt &new_expr);

  virtual bool get_var(const clang::VarDecl &vd, exprt &new_expr);

  virtual bool get_function(const clang::FunctionDecl &fd, exprt &new_expr);

  virtual bool
  get_function_body(const clang::FunctionDecl &fd, exprt &new_expr);

  /**
   *  Parse function parameters
   *  This function simply contains a loop to populate the code argument list
   *  and calls get_function_body to parse each individual parameter.
   */
  virtual bool get_function_params(
    const clang::FunctionDecl &fd,
    code_typet::argumentst &params);

  /**
   *  Parse each individual parameter of the function
   */
  bool get_function_param(const clang::ParmVarDecl &pd, exprt &param);

  virtual bool get_struct_union_class(const clang::RecordDecl &recordd);

  virtual bool get_struct_union_class_fields(
    const clang::RecordDecl &recordd,
    struct_union_typet &type);

  virtual bool get_struct_union_class_methods(
    const clang::RecordDecl &recordd,
    struct_union_typet &type);

  virtual bool get_type(const clang::QualType &type, typet &new_type);

  virtual bool get_type(const clang::Type &the_type, typet &new_type);

  bool get_builtin_type(const clang::BuiltinType &bt, typet &new_type);

  virtual bool get_expr(const clang::Stmt &stmt, exprt &new_expr);

  bool get_decl_ref(const clang::Decl &decl, exprt &new_expr);

  bool
  get_binary_operator_expr(const clang::BinaryOperator &binop, exprt &new_expr);

  bool get_compound_assign_expr(
    const clang::CompoundAssignOperator &compop,
    exprt &new_expr);

  bool
  get_unary_operator_expr(const clang::UnaryOperator &uniop, exprt &new_expr);

  bool get_atomic_expr(const clang::AtomicExpr &atm, exprt &new_expr);

  bool get_cast_expr(const clang::CastExpr &cast, exprt &new_expr);

  void get_default_symbol(
    symbolt &symbol,
    irep_idt module_name,
    typet type,
    irep_idt base_name,
    irep_idt unique_name,
    locationt location);

  void
  get_decl_name(const clang::NamedDecl &vd, std::string &id, std::string &name);

  void
  get_start_location_from_stmt(const clang::Stmt &stmt, locationt &location);

  void
  get_final_location_from_stmt(const clang::Stmt &stmt, locationt &location);

  void get_location_from_decl(const clang::Decl &decl, locationt &location);

  void set_location(
    clang::PresumedLoc &PLoc,
    std::string &function_name,
    locationt &location);

  void get_presumed_location(
    const clang::SourceLocation &loc,
    clang::PresumedLoc &PLoc);

  std::string get_filename_from_path(std::string path);
  std::string get_modulename_from_path(std::string path);

  void convert_expression_to_code(exprt &expr);

  symbolt *move_symbol_to_context(symbolt &symbol);

  bool convert_character_literal(
    const clang::CharacterLiteral &char_literal,
    exprt &dest);

  bool convert_string_literal(
    const clang::StringLiteral &string_literal,
    exprt &dest);

  bool convert_integer_literal(
    const clang::IntegerLiteral &integer_literal,
    exprt &dest);

  bool convert_float_literal(
    const clang::FloatingLiteral &floating_literal,
    exprt &dest);

  const clang::Decl *get_DeclContext_from_Stmt(const clang::Stmt &stmt);

  const clang::Decl *get_top_FunctionDecl_from_Stmt(const clang::Stmt &stmt);
};

#endif /* CLANG_C_FRONTEND_CLANG_C_CONVERT_H_ */
