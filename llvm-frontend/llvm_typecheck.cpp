/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_typecheck.h"

#include <std_types.h>
#include <ansi-c/c_types.h>

llvm_typecheckt::llvm_typecheckt(contextt &_context)
  : context(_context)
{
}

llvm_typecheckt::~llvm_typecheckt()
{
}

bool llvm_typecheckt::typecheck()
{
  if(convert_top_level_decl())
    return true;

  return false;
}

bool llvm_typecheckt::convert_top_level_decl()
{
  // Iterate through each translation unit and their global symbols, creating
  // symbols as we go.

  for (auto &translation_unit : ASTs) {
    clang::ASTUnit::top_level_iterator it;
    for (it = translation_unit->top_level_begin();
        it != translation_unit->top_level_end(); it++) {
      switch ((*it)->getKind()) {
        case clang::Decl::Typedef:
        case clang::Decl::Function:
        case clang::Decl::Record:
        case clang::Decl::Var:
        default:
          std::cerr << "Unrecognized / unimplemented decl type ";
          std::cerr << (*it)->getDeclKindName() << std::endl;
          abort();
      }
    }
  }

  return false;
}

void llvm_typecheckt::get_type(const clang::Type &the_type, typet &new_type)
{
  clang::Type::TypeClass tc = the_type.getTypeClass();
  switch (tc) {
    case clang::Type::Builtin:
    {
      const clang::BuiltinType &bt = static_cast<const clang::BuiltinType&>(the_type);
      switch (bt.getKind()) {
        case clang::BuiltinType::Void:
          abort();
          break;

        case clang::BuiltinType::Bool:
          new_type = bool_type();
          break;

        case clang::BuiltinType::UChar:
          new_type = unsignedbv_typet(config.ansi_c.char_width);
          break;

        case clang::BuiltinType::Char16:
          new_type = unsignedbv_typet(16);
          break;

        case clang::BuiltinType::Char32:
          new_type = unsignedbv_typet(32);
          break;

        case clang::BuiltinType::UShort:
          new_type = unsignedbv_typet(config.ansi_c.short_int_width);
          break;

        case clang::BuiltinType::UInt:
          new_type = uint_type();
          break;

        case clang::BuiltinType::ULong:
          new_type = long_uint_type();
          break;

        case clang::BuiltinType::ULongLong:
          new_type = long_long_uint_type();
          break;

        case clang::BuiltinType::UInt128:
          // Various simplification / big-int related things use uint64_t's...
          std::cerr << "No support for uint128's in ESBMC right now, sorry" << std::endl;
          abort();

        case clang::BuiltinType::SChar:
          new_type = signedbv_typet(config.ansi_c.char_width);
          break;

        case clang::BuiltinType::Short:
          new_type = signedbv_typet(config.ansi_c.short_int_width);
          break;

        case clang::BuiltinType::Int:
          new_type = int_type();
          break;

        case clang::BuiltinType::Long:
          new_type = long_int_type();
          break;

        case clang::BuiltinType::LongLong:
          new_type = long_long_int_type();
          break;

        case clang::BuiltinType::Int128:
          // Various simplification / big-int related things use uint64_t's...
          std::cerr << "No support for uint128's in ESBMC right now, sorry" << std::endl;
          abort();

        case clang::BuiltinType::Float:
          new_type = float_type();
          break;

        case clang::BuiltinType::Double:
          new_type = double_type();
          break;

        case clang::BuiltinType::LongDouble:
          new_type = long_double_type();
          break;

        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::WChar_S:
        case clang::BuiltinType::WChar_U:
        case clang::BuiltinType::NullPtr:
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCClass:
        case clang::BuiltinType::ObjCSel:
        case clang::BuiltinType::OCLImage1d:
        case clang::BuiltinType::OCLImage1dArray:
        case clang::BuiltinType::OCLImage1dBuffer:
        case clang::BuiltinType::OCLImage2d:
        case clang::BuiltinType::OCLImage2dArray:
        case clang::BuiltinType::OCLImage3d:
        case clang::BuiltinType::OCLSampler:
        case clang::BuiltinType::OCLEvent:
        case clang::BuiltinType::Dependent:
        case clang::BuiltinType::Overload:
        case clang::BuiltinType::BoundMember:
        case clang::BuiltinType::PseudoObject:
        case clang::BuiltinType::UnknownAny:
        case clang::BuiltinType::BuiltinFn:
        case clang::BuiltinType::ARCUnbridgedCast:
        case clang::BuiltinType::Half:
          std::cerr << "Unrecognized clang builtin type "
                    << bt.getName(clang::PrintingPolicy(clang::LangOptions())).str()
                    << std::endl;
          abort();
      }
    }
    return;

    case clang::Type::Record:
      return;

    case clang::Type::ConstantArray:
      return;

    case clang::Type::Elaborated:
      return;

    case clang::Type::Pointer:
      return;

    case clang::Type::Typedef:
      return;

    case clang::Type::FunctionProto:
      return;

    case clang::Type::FunctionNoProto:
      return;

    case clang::Type::IncompleteArray:
      return;

    case clang::Type::Paren:
      return;

    default:
      std::cerr << "No clang <=> ESBMC migration for type "
                << the_type.getTypeClassName() << std::endl;
      abort();
  }
}
