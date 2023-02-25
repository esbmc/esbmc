#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclFriend.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/QualTypeNames.h>
#include <clang/AST/Type.h>
#include <clang/Index/USRGeneration.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/AST/ParentMapContext.h>
#include <llvm/Support/raw_os_ostream.h>
CC_DIAGNOSTIC_POP()

#include <clang-cpp-frontend/clang_cpp_convert.h>

bool clang_cpp_convertert::get_struct_class_virtual_methods(
  const clang::CXXRecordDecl *cxxrd,
  struct_typet &type)
{
  for(const auto &md : cxxrd->methods())
  {
    if(!md->isVirtual())
      continue;

    // TODO: take care of overriding methods
    assert(md->begin_overridden_methods() == md->end_overridden_methods());

    /*
     * 1. convert this virtual method and add them to class symbol type
     */
    struct_typet::componentt comp;
    if(get_decl(*md, comp))
      return true;

    if(annotate_virtual_overriding_methods(md, comp))
      return true;
    type.methods().push_back(comp);

    /*
     * 2. If this is the first time we see a virtual method in this class,
     *  add virtual table type symbol and virtual pointer.
     *  Otherwise just add an entry in the existing virtual table type symbol
     */
    if(!check_vtable_existence(type))
    {
      if(add_vtable_type_symbol(md, type))
        return true;
      if(add_vptr(md, type))
        return true;
    }
    else if(add_vtable_type_entry(type, comp))
      return true;
  }

  assert(!"TODO: check tag.Bird type, vtable type and vptr");

  return false;
}

bool clang_cpp_convertert::annotate_virtual_overriding_methods(
  const clang::CXXMethodDecl *md,
  struct_typet::componentt &comp)
{
  // TODO: take care of overriding methods
  assert(md->begin_overridden_methods() == md->end_overridden_methods());

  comp.type().set("#is_virtual", true);
  comp.type().set("#virtual_name", comp.name().as_string());
  comp.set("#is_virtual", true);
  comp.set("#virtual_name", comp.name().as_string());

  return false;
}

bool clang_cpp_convertert::check_vtable_existence(struct_typet &type)
{
  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();
  symbolt *s = context.find_symbol(vt_name);
  // first time we create the vtable type for this class
  if(s == nullptr)
    return false; // vtable type symbol doesn't exist

  return true; // vtable type symbol exists
}

bool clang_cpp_convertert::add_vtable_type_symbol(
  const clang::CXXMethodDecl *md,
  struct_typet &type)
{
  /*
   *  We model the virtual function table as struct:
   *   typedef struct {
   *      void (*do_something)(Base*);
   *    } VftTag_Base;
   * Later, we will instantiate the virtual table as:
   *  VftTag_Base vtable_TagBase@TagBase = { .do_something = &TagBase::do_someting(); }
   */
  assert(!"TODO: First add vtable");
  return false;
}

bool clang_cpp_convertert::add_vptr(
  const clang::CXXMethodDecl *md,
  struct_typet &type)
{
  assert(!"TODO: add vptr");
  return false;
}

bool clang_cpp_convertert::add_vtable_type_entry(
  struct_typet &type,
  struct_typet::componentt &comp)
{
  assert(!"TODO: Add entry in existing vtable type");
  return false;
}
