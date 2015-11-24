/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_convert.h"

#include <arith_tools.h>
#include <bitvector.h>

#include <ansi-c/c_types.h>

void llvm_convertert::convert_character_literal(
  const clang::CharacterLiteral char_literal,
  exprt &dest)
{
  typet type = char_type();
  switch(char_literal.getKind())
  {
    case clang::CharacterLiteral::Ascii:
      type.set("#c_type", "char");
      break;

    case clang::CharacterLiteral::UTF16:
      type.width(16);
      type.set("#c_type", "char16_t");
      break;

    case clang::CharacterLiteral::UTF32:
    case clang::CharacterLiteral::Wide:
      type.width(32);
      type.set("#c_type", "char32_t");
      break;

    default:
      std::cerr << "Conversion of unsupported char literal: \"";
      char_literal.dump();
      abort();
  }

  dest =
    constant_exprt(integer2binary(char_literal.getValue(), bv_width(type)), type);
}
