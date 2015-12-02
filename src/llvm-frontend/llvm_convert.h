/*
 * llvmtypecheck.h
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_CONVERT_H_
#define LLVM_FRONTEND_LLVM_CONVERT_H_

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <context.h>
#include <namespace.h>
#include <std_types.h>

#include <clang/Frontend/ASTUnit.h>
#include <clang/AST/Type.h>
#include <clang/AST/Expr.h>

class llvm_convertert
{
public:
  llvm_convertert(
    contextt &_context,
    std::vector<std::unique_ptr<clang::ASTUnit> > &_ASTs);
  virtual ~llvm_convertert();

  bool convert();

private:
  contextt &context;
  namespacet ns;
  std::vector<std::unique_ptr<clang::ASTUnit> > &ASTs;

  locationt current_location;
  std::string current_path;

  // TODO: We don't need the current_function_name attribute!!!!!
  // We can get it from the declcontext. Refactor the
  // rest of the code to use it
  std::string current_function_name;
  unsigned int current_scope_var_num;

  unsigned int anon_counter;

  clang::SourceManager *sm;

  typedef std::map<std::size_t, std::string> object_mapt;
  object_mapt object_map;

  typedef std::map<std::size_t, std::string> type_mapt;
  type_mapt type_map;

  void dump_type_map();
  void dump_object_map();

  bool convert_builtin_types();
  bool convert_top_level_decl();

  void get_decl(
    const clang::Decl &decl,
    exprt &new_expr);

  void get_typedef(
    const clang::TypedefDecl &tdd,
    exprt &new_expr);

  void get_var(
    const clang::VarDecl &vd,
    exprt &new_expr);

  void get_function(
    const clang::FunctionDecl &fd,
    exprt &new_expr);

  void get_function_params(
    const clang::ParmVarDecl &pdecl,
    exprt &param);

  void get_enum(
    const clang::EnumDecl &enumd,
    exprt &new_expr);

  void get_struct_union_class(
    const clang::RecordDecl &recordd,
    exprt &new_expr);

  void get_enum_constants(
    const clang::EnumConstantDecl &enumcd,
    exprt &new_expr);

  void get_type(
    const clang::QualType &the_type,
    typet &new_type);

  void get_builtin_type(
    const clang::BuiltinType &bt,
    typet &new_type);

  void get_expr(
    const clang::Stmt &stmt,
    exprt &new_expr);

  void get_decl_ref(
    const clang::Decl &decl,
    exprt &new_expr);

  void get_binary_operator_expr(
    const clang::BinaryOperator &binop,
    exprt &new_expr);

  void get_unary_operator_expr(
    const clang::UnaryOperator &uniop,
    exprt &new_expr);

  void get_cast_expr(
    const clang::CastExpr &cast,
    exprt &new_expr);

  void get_predefined_expr(
    const clang::PredefinedExpr &pred_expr,
    exprt &new_expr);

  void get_default_symbol(
    symbolt &symbol,
    typet type,
    std::string base_name,
    std::string pretty_name,
    bool is_local);

  std::string get_var_name(
    std::string name,
    bool is_local);

  std::string get_param_name(
    std::string name);

  std::string get_tag_name(
    std::string _name);

  void set_source_manager(clang::SourceManager &source_manager);
  void update_current_location(clang::SourceLocation source_location);
  std::string get_filename_from_path();
  std::string get_modulename_from_path();

  void convert_expression_to_code(exprt& expr);

  void move_symbol_to_context(symbolt &symbol);

  void check_symbol_redefinition(
    symbolt &old_symbol,
    symbolt &new_symbol);

  void convert_character_literal(
    const clang::CharacterLiteral char_literal,
    exprt &dest);

  void convert_string_literal(
    const clang::StringLiteral string_literal,
    exprt &dest);

  void convert_integer_literal(
    llvm::APInt val,
    typet type,
    exprt &dest);

  void convert_float_literal(
    llvm::APFloat val,
    typet type,
    exprt &dest);

  void parse_float(
    const std::string &src,
    mp_integer &significand,
    mp_integer &exponent, // base 10
    bool &is_float,
    bool &is_long);
};

#endif /* LLVM_FRONTEND_LLVM_CONVERT_H_ */
