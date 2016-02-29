/*
 * llvm_main.h
 *
 *  Created on: Jul 29, 2015
 *      Author: mramalho
 */

#ifndef LLVM_FRONTEND_LLVM_MAIN_H_
#define LLVM_FRONTEND_LLVM_MAIN_H_

#include <context.h>
#include <message.h>
#include <std_code.h>

bool llvm_main(
  contextt &context,
  const std::string &default_prefix,
  const std::string &standard_main,
  message_handlert &message_handler);

#endif /* LLVM_FRONTEND_LLVM_MAIN_H_ */
