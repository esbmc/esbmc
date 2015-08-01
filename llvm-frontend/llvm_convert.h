/*
 * llvmtypecheck.h
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_CONVERT_H_
#define LLVM_FRONTEND_LLVM_CONVERT_H_

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
  int current_scope;
  clang::SourceManager *sm;

  bool convert_top_level_decl();

  void convert_decl(
    const clang::Decl &decl,
    exprt &new_expr);

  void convert_typedef(
    const clang::TypedefDecl &tdd,
    exprt &new_expr);

  void convert_var(
    const clang::VarDecl &vd,
    exprt &new_expr);

  void convert_function(const clang::FunctionDecl &fd);
  code_typet::argumentt convert_function_params(
    std::string function_name,
    clang::ParmVarDecl *pdecl);

  void get_type(
    const clang::QualType &the_type,
    typet &new_type);

  void get_builtin_type(
    const clang::BuiltinType &bt,
    typet &new_type);

  void get_expr(
    const clang::Stmt &stmt,
    exprt &new_expr);

  void get_decl_expr(
    const clang::Decl &decl,
    exprt &new_expr);

  void get_binary_operator_expr(
    const clang::BinaryOperator &binop,
    exprt &new_expr);

  void get_cast_expr(
    const clang::CastExpr &cast,
    exprt &new_expr);

  void gen_typecast(
    exprt &expr,
    typet type);

  void get_default_symbol(symbolt &symbol);
  std::string get_var_name(std::string name, bool is_local);
  std::string get_param_name(std::string name);

  void set_source_manager(clang::SourceManager &source_manager);
  void update_current_location(clang::SourceLocation source_location);
  std::string get_filename_from_path();
  std::string get_modulename_from_path();
};

#endif /* LLVM_FRONTEND_LLVM_CONVERT_H_ */
