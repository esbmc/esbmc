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
  llvm_convertert(contextt &_context);
  virtual ~llvm_convertert();

  bool convert();
  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;

private:
  contextt &context;
  namespacet ns;
  locationt current_location;
  std::string current_path;
  std::string current_function_name;
  int current_scope_var_num;
  clang::SourceManager *sm;

  typedef std::map<int, std::string> object_mapst;
  object_mapst object_map;

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
    const clang::FunctionDecl &fd);

  void get_function_params(
    const clang::ParmVarDecl &pdecl,
    exprt &param);

  void get_enum(
    const clang::EnumDecl &enumd,
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

  void gen_typecast(
    exprt &expr,
    typet type);

  void get_default_symbol(
    symbolt &symbol,
    typet type,
    std::string base_name,
    std::string pretty_name);

  std::string get_var_name(
    std::string name,
    bool is_local);

  std::string get_param_name(
    std::string name);

  std::string get_tag_name(
    std::string name,
    bool is_local);

  void get_size_exprt(
    llvm::APInt val,
    typet type,
    exprt &expr);

  void get_size_exprt(
    double val,
    typet type,
    exprt &expr);

  void set_source_manager(clang::SourceManager &source_manager);
  void update_current_location(clang::SourceLocation source_location);
  std::string get_filename_from_path();
  std::string get_modulename_from_path();

  void move_symbol_to_context(symbolt &symbol);

  void check_symbol_redefinition(
    symbolt &old_symbol,
    symbolt &new_symbol);

  void convert_exprt_inside_block(
    exprt &expr);
};

#endif /* LLVM_FRONTEND_LLVM_CONVERT_H_ */
