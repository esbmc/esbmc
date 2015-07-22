/*
 * llvmparsert.h
 *
 *  Created on: Jul 22, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_PARSER_H_
#define LLVM_FRONTEND_LLVM_PARSER_H_

class llvm_parsert
{
public:
  irep_idt filename;

  llvm_parsert();
  virtual ~llvm_parsert();

  bool parse();
};

extern llvm_parsert llvm_parser;

#endif /* LLVM_FRONTEND_LLVM_PARSER_H_ */
