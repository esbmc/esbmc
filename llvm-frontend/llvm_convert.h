/*
 * llvmtypecheck.h
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_CONVERT_H_
#define LLVM_FRONTEND_LLVM_CONVERT_H_

#include <context.h>

#include <clang/Frontend/ASTUnit.h>

class llvm_convertert
{
public:
  llvm_convertert(contextt &_context);

  virtual ~llvm_convertert();

  bool convert();

  std::vector<std::unique_ptr<clang::ASTUnit> > ASTs;

private:
  contextt &context;

  bool convert_top_level_decl();

  void get_type(const clang::Type &the_type, typet &new_type);
  void get_default_symbol(symbolt &symbol, clang::ASTUnit::top_level_iterator it);

  std::string get_filename_from_path(std::string path);
  std::string get_modulename_from_path(std::string path);
};

#endif /* LLVM_FRONTEND_LLVM_CONVERT_H_ */
