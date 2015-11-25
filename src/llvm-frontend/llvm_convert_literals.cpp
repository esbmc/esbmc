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
#include <ansi-c/convert_integer_literal.h>

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

void llvm_convertert::convert_string_literal(
  const clang::StringLiteral string_literal,
  exprt &dest)
{
  constant_exprt string(string_literal.getBytes().str(), array_typet());

  array_typet &type = to_array_type(string.type());
  typet &elem_type = type.subtype();

  switch(string_literal.getKind())
  {
    case clang::StringLiteral::Ascii:
    case clang::StringLiteral::UTF8:
      elem_type = char_type();
      elem_type.set("#c_type", "char");
      break;

    case clang::StringLiteral::UTF16:
      elem_type = char16_type();
      elem_type.set("#c_type", "char16_t");
      break;

    case clang::StringLiteral::UTF32:
      elem_type = char32_type();
      elem_type.set("#c_type", "char32_t");
      break;

    case clang::StringLiteral::Wide:
      elem_type = wchar_type();
      elem_type.set("#c_type", "wchar_t");
      break;

    default:
      std::cerr << "Conversion of unsupported char literal: \"";
      string_literal.dump();
      abort();
  }

  exprt& size = to_array_type(type).size();
  convert_integer_literal(integer2string(string_literal.getLength()+1), size);

  for(u_int byte = 0; byte < string_literal.getLength(); ++byte)
  {
    exprt elem =
      constant_exprt(
        integer2binary(string_literal.getCodeUnit(byte),
        bv_width(elem_type)),
        elem_type);
    elem.set("#cformat", string_literal.getCodeUnit(byte));

    string.operands().push_back(elem);
  }

  dest.swap(string);
}

void llvm_convertert::convert_integer_literal(
  const llvm::APInt val,
  typet type,
  exprt &dest)
{
  assert(type.is_unsignedbv() || type.is_signedbv());

  exprt the_val;
  if (type.is_unsignedbv())
  {
    the_val =
      constant_exprt(
        integer2binary(val.getZExtValue(),
        bv_width(type)),
        type);
    the_val.set("#cformat", val.getZExtValue());
  }
  else
  {
    the_val =
      constant_exprt(
        integer2binary(val.getSExtValue(),
        bv_width(type)),
        type);
    the_val.set("#cformat", val.getSExtValue());
  }

  dest.swap(the_val);
}
