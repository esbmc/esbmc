/*
 * llvmtypecheck.h
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_CONVERT_H_
#define LLVM_FRONTEND_LLVM_CONVERT_H_

#include <context.h>
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
  locationt current_location;
  std::string current_path;

  bool convert_top_level_decl();

  void convert_decl(
    unsigned int scope_number,
    std::string function_name,
    const clang::Decl &decl,
    exprt &new_expr);

  void convert_decl(
    const clang::Decl &decl,
    exprt &new_expr);

  void convert_typedef(
    unsigned int scope_number,
    std::string function_name,
    const clang::TypedefDecl &tdd,
    exprt &new_expr);

  void convert_typedef(
    const clang::TypedefDecl &tdd,
    exprt &new_expr);

  void convert_var(
    unsigned int scope_number,
    std::string function_name,
    const clang::VarDecl &vd,
    exprt &new_expr);

  void convert_var(
    const clang::VarDecl &vd,
    exprt &new_expr);

  void convert_function(const clang::FunctionDecl &fd);
  code_typet::argumentt convert_function_params(
    std::string function_name,
    clang::ParmVarDecl *pdecl);

  void get_type(const clang::QualType &the_type, typet &new_type);
  void get_expr(const clang::Stmt &expr, exprt &new_expr);
  void get_default_symbol(symbolt &symbol);

  void update_current_location(clang::ASTUnit::top_level_iterator it);
  std::string get_filename_from_path();
  std::string get_modulename_from_path();
};

#endif /* LLVM_FRONTEND_LLVM_CONVERT_H_ */
