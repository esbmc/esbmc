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

    /*
     * 1. convert this virtual method and add them to class symbol type
     */
    struct_typet::componentt comp;
    if(get_decl(*md, comp))
      return true;

    // additional annotations for virtual/overriding methods
    if(annotate_virtual_overriding_methods(md, comp))
      return true;
    type.methods().push_back(comp);

    /*
     * 2. If this is the first time we see a virtual method in this class,
     *  add virtual table type symbol and virtual pointer. Then add a new
     *  entry in the vtable.
     */
    symbolt *vtable_type_symbol = check_vtable_type_symbol_existence(type);
    if(!vtable_type_symbol)
    {
      // first time we create the vtable type for this class
      vtable_type_symbol = add_vtable_type_symbol(md, comp, type);
      if(vtable_type_symbol == nullptr)
        return true;

      add_vptr(md, type);
    }

    /*
     * 3. add an entry in the existing virtual table type symbol
     */
    add_vtable_type_entry(type, comp, vtable_type_symbol);

    /*
     * 4. deal with overriding method
     */
    if(md->begin_overridden_methods() != md->end_overridden_methods())
    {
      /*
       * Assume it *always* points to one overriden method in base class
       * TODO: Use a loop if there are more overriden methods
       *    md->overriden_methods() should do the job
       */
      assert(md->size_overridden_methods() == 1);
      add_thunk_method(md->begin_overridden_methods(), type);
    }
  }

  return false;
}

bool clang_cpp_convertert::annotate_virtual_overriding_methods(
  const clang::CXXMethodDecl *md,
  struct_typet::componentt &comp)
{
  // TODO: take care of overriding methods
  //assert(md->begin_overridden_methods() == md->end_overridden_methods());

  comp.type().set("#is_virtual", true);
  comp.type().set("#virtual_name", comp.name().as_string());
  comp.set("is_virtual", true);
  comp.set("virtual_name", comp.name().as_string());

  return false;
}

symbolt *
clang_cpp_convertert::check_vtable_type_symbol_existence(struct_typet &type)
{
  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();
  return context.find_symbol(vt_name);
}

symbolt *clang_cpp_convertert::add_vtable_type_symbol(
  const clang::CXXMethodDecl *md,
  const struct_typet::componentt &comp,
  struct_typet &type)
{
  /*
   *  We model the type of the virtual table as a struct type, something like:
   *   typedef struct {
   *      void (*do_something)(Base*);
   *    } VftTag_Base;
   * Later, we will instantiate a virtual table as:
   *  VftTag_Base vtable_TagBase@TagBase = { .do_something = &TagBase::do_someting(); }
   *
   *  Vtable type has the id in the form of `virtual_table::tag-BLAH`.
   */

  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();

  symbolt vt_type_symb;
  vt_type_symb.id = vt_name;
  vt_type_symb.name = vtable_type_prefix + type.tag().as_string();
  vt_type_symb.mode = mode;
  vt_type_symb.type = struct_typet();
  vt_type_symb.is_type = true;
  vt_type_symb.type.set("name", vt_type_symb.id);
  vt_type_symb.location = comp.location();
  vt_type_symb.module =
    get_modulename_from_path(comp.location().file().as_string());

  if(context.move(vt_type_symb))
  {
    log_error(
      "Couldn't add vtable type symbol {} to symbol table", vt_type_symb.id);
    abort();
  }

  return context.find_symbol(vt_name);
}

void clang_cpp_convertert::add_vptr(
  const clang::CXXMethodDecl *md,
  struct_typet &type)
{
  /*
   * We model the virtual pointer as a `component` to the parent class' type.
   * This will be the vptr pointing to the vtable that contains the overriden functions.
   *
   * Vptr has the name in the form of `tag-BLAH@vtable_pointer`.
   */

  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();
  // add a virtual-table pointer
  struct_typet::componentt component;
  component.type() = pointer_typet(symbol_typet(vt_name));
  component.set_name(tag_prefix + type.tag().as_string() + "::@vtable_pointer");
  component.base_name("@vtable_pointer");
  component.pretty_name(type.tag().as_string() + "@vtable_pointer");
  component.set("is_vtptr", true);
  component.set("access", "public");
  // add to the class' type
  type.components().push_back(component);
}

void clang_cpp_convertert::add_vtable_type_entry(
  struct_typet &type,
  struct_typet::componentt &comp,
  symbolt *vtable_type_symbol)
{
  /*
   * When we encounter a virtual or overriding method in a class,
   * need to add an entry to the vtable type symbol.
   * Since the vtable type symbol is modelled as a struct,
   * this entry is considered a `component` in this struct.
   * We model this entry as a function pointer, pointing to the
   * virtual or overriding method in this class.
   *
   * Vtable entry's name is of the form ``virtual_table::tag.BLAH::do_something().
   */

  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();
  std::string virtual_name = comp.name().as_string();
  struct_typet::componentt vt_entry;
  vt_entry.type() = pointer_typet(comp.type());
  vt_entry.set_name(vt_name.as_string() + "::" + virtual_name);
  vt_entry.set("base_name", virtual_name);
  vt_entry.set("pretty_name", virtual_name);
  vt_entry.set("access", "public");
  vt_entry.location() = comp.location();
  // add an entry to the virtual table
  assert(vtable_type_symbol);
  struct_typet &vtable_type = to_struct_type(vtable_type_symbol->type);
  vtable_type.components().push_back(vt_entry);
}

void clang_cpp_convertert::add_thunk_method(
  const clang::CXXMethodDecl *const *md,
  struct_typet &type)
{
  /*
   * Add a thunk function for a overriding method.
   * This thunk function will be added as a symbol in the symbol table,
   * and considered a `component` to the derived class' type.
   * This thunk function will be used to set up the derived class' vtable
   * to override the base method, e.g.
   *
   * Suppose Penguin derives Bird, we have the following vtables for Penguin:
   *  virtual_table::Bird@Penguin =
   *    {
   *      .do_it() = &Penguin::do_it()::tag.Bird; // this is the thunk redirecting call to the overriding function
   *    };
   *
   *  virtual_table::Penguin@Penguin =
   *    {
   *      .do_it() = &Penguin::do_it(); // this is the overriding function
   *    };
   */

  assert(!"TODO: add thunks");
}
