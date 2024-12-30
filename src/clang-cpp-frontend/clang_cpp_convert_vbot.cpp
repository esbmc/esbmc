
#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Basic/Version.inc>
#include <clang/AST/Attr.h>
#include "clang/AST/CXXInheritance.h"
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
#include <util/message.h>
#include <util/expr_util.h>
#include "util/c_types.h"
#include "util/cpp_base_offset.h"
#include "util/cpp_data_object.h"
#include "util/simplify_expr.h"

bool clang_cpp_convertert::get_struct_class_virtual_base_offsets(
  const clang::CXXRecordDecl &cxxrd,
  struct_typet &type)
{
  // Add a vbase_offset pointer if this type has any virtual bases
  if (cxxrd.getNumVBases() > 0)
  {
    symbolt *vbase_offset_type_symbol =
      add_vbase_offset_type_symbol_and_vbase_offset_ptr_if_needed(
        type, type.location());
    if (!vbase_offset_type_symbol)
      return true;

    for (const auto &vb : cxxrd.vbases())
    {
      const clang::CXXRecordDecl *vb_rd = vb.getType()->getAsCXXRecordDecl();
      std::string decl_id, decl_name;
      get_decl_name(*vb_rd, decl_name, decl_id);
      struct_typet::componentt comp;
      comp.type() = size_type();
      comp.set_name(decl_name);
      comp.pretty_name(decl_name);
      comp.base_name(decl_name);
      /*
       * Add an entry in the existing virtual base offset table type symbol
       */
      add_vbase_offset_table_type_entry(type, comp, vbase_offset_type_symbol);
    }
  }
  return false;
}

symbolt *clang_cpp_convertert::
  add_vbase_offset_type_symbol_and_vbase_offset_ptr_if_needed(
    struct_typet &type,
    const locationt &loc)
{
  symbolt *vbase_offset_type_symbol =
    check_vbase_offset_type_symbol_existence(type);
  if (!vbase_offset_type_symbol)
  {
    // first time we create the vbase_offset type for this class
    vbase_offset_type_symbol = add_vbase_offset_type_symbol(loc, type);
    if (vbase_offset_type_symbol == nullptr)
      return nullptr;

    add_vbase_offset_ptr(type);
  }
  return vbase_offset_type_symbol;
}
symbolt *clang_cpp_convertert::check_vbase_offset_type_symbol_existence(
  const struct_typet &type)
{
  irep_idt vbase_offset_type_name =
    vbase_offset_type_prefix + tag_prefix + type.tag().as_string();
  return context.find_symbol(vbase_offset_type_name);
}
symbolt *clang_cpp_convertert::add_vbase_offset_type_symbol(
  const locationt &loc,
  struct_typet &type)
{
  /*
   *  We model the type of the virtual base offset as a struct type, something like:
   *   typedef struct {
   *      size_t offset_to_base_x;
   *    } VbaseOffsets_Tag_Base;
   * Later, we will instantiate a virtual base offset table as:
   *  VbaseOffsets_Tag_Base vbase_offset_table_TagBase@TagBase = { .offset_to_base_x = offset; }
   *
   *  Vtable type has the id in the form of `vbase_offset_table::tag-BLAH`.
   */
  assert(!type.tag().empty());

  irep_idt vt_name =
    vbase_offset_type_prefix + tag_prefix + type.tag().as_string();

  symbolt vbo_table;
  vbo_table.id = vt_name;
  vbo_table.name = vbase_offset_type_prefix + type.tag().as_string();
  vbo_table.mode = mode;
  vbo_table.type = struct_typet();
  vbo_table.is_type = true;
  vbo_table.type.set("name", vbo_table.id);
  vbo_table.location = loc;
  vbo_table.module = get_modulename_from_path(loc.file().as_string());

  if (context.move(vbo_table))
  {
    log_error(
      "Failed add virtual base offset type symbol {} to symbol table",
      vbo_table.id);
    abort();
  }

  return context.find_symbol(vt_name);
}
void clang_cpp_convertert::add_vbase_offset_ptr(struct_typet &type)
{
  /*
   * We model the virtual base offset pointer as a `component` to the parent class' type.
   * This will be the vbo_ptr pointing to the vbo_table that contains the offset to the virtual bases.
   *
   * Vbo_ptr has the name in the form of `tag-BLAH@vbase_offset_ptr`, where BLAH is the class name.
   */
  struct_typet &data_object_type = cpp_data_object::get_data_object_type(
    tag_prefix + type.tag().as_string(), context);
  assert(has_suffix(
    data_object_type.tag().as_string(), cpp_data_object::data_object_suffix));

  irep_idt vbo_name =
    vbase_offset_type_prefix + tag_prefix + type.tag().as_string();
  // add a virtual-table pointer
  struct_typet::componentt component;
  component.type() = pointer_typet(symbol_typet(vbo_name));
  component.set_name(
    tag_prefix + type.tag().as_string() + "::" + vbase_offset_ptr_suffix);
  component.base_name(vbase_offset_ptr_suffix);
  component.pretty_name(type.tag().as_string() + vbase_offset_ptr_suffix);
  component.set("is_vbot_ptr", true);
  component.set("access", "public");
  // add to the class' type
  data_object_type.components().push_back(component);

  type.set("#has_vbot_ptr_component", true);
}
void clang_cpp_convertert::add_vbase_offset_table_type_entry(
  struct_typet &type,
  struct_typet::componentt &comp,
  symbolt *vbase_offset_table_type_symbol)
{
  std::string virtual_name = comp.name().as_string();
  struct_typet::componentt vbot_entry;
  vbot_entry.type() = comp.type();
  vbot_entry.set_name(comp.base_name());
  vbot_entry.set("base_name", comp.base_name());
  vbot_entry.set("pretty_name", comp.get("pretty_name"));
  vbot_entry.set("access", "public");
  vbot_entry.location() = comp.location();
  // add an entry to the virtual table
  assert(vbase_offset_table_type_symbol);
  struct_typet &vbot_type =
    to_struct_type(vbase_offset_table_type_symbol->type);
  vbot_type.components().push_back(vbot_entry);
}
void clang_cpp_convertert::setup_vbo_table_struct_variables(
  const clang::CXXRecordDecl &cxxrd,
  const struct_typet &type)
{
  /*
   * We model the virtual base offset table (vbot) as
   * a struct of offsets.
   */
  if (cxxrd.getNumVBases() > 0)
  {
    add_vbo_table_variable_symbols(cxxrd, type);
  }
}

void clang_cpp_convertert::add_vbo_table_variable_symbols(
  const clang::CXXRecordDecl &cxxrd,
  const struct_typet &type)
{
  // This is the class we are currently dealing with
  std::string class_id, class_name;
  get_decl_name(cxxrd, class_name, class_id);

  // Find all bases that have a virtual base themselves
  std::set<const clang::CXXRecordDecl *> bases_with_vbases;
  cxxrd.forallBases([&bases_with_vbases](const clang::CXXRecordDecl *base) {
    if (base->getNumVBases() > 0)
    {
      bases_with_vbases.insert(base);
    }
    return true;
  });

  for (const auto &base : bases_with_vbases)
  {
    clang::CXXBasePaths paths;
    cxxrd.lookupInBases(
      [&base](
        const clang::CXXBaseSpecifier *specifier, clang::CXXBasePath &path) {
        return specifier->getType()->getAsCXXRecordDecl() == base;
      },
      paths);
    for (const auto &path : paths)
      handle_base_with_path(type, class_id, *base, path);
  }

  // This class also has a vbase even if itself is not a base.
  assert(cxxrd.getNumVBases() > 0);
  // The path to the class itself is empty
  clang::CXXBasePath path_to_class = clang::CXXBasePath();
  handle_base_with_path(type, class_id, cxxrd, path_to_class);
}
void clang_cpp_convertert::handle_base_with_path(
  const struct_typet &type,
  const std::string &class_id,
  const clang::CXXRecordDecl &base,
  const clang::CXXBasePath &path)
{
  std::string base_path_id;
  std::string base_class_name, base_class_id;
  {
    // This is the base class we are currently dealing with

    get_decl_name(base, base_class_name, base_class_id);
    assert(!base_class_id.empty());

    base_path_id += base_class_id;
    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
      // Ignore the last element in the path as its id is equal to the class_id
      if (std::next(it) == path.rend())
        break;
      const auto &specifier = *it;
      const clang::CXXRecordDecl *vb_rd = specifier.Class;
      std::string decl_id, decl_name;
      get_decl_name(*vb_rd, decl_name, decl_id);
      base_path_id += "@";
      base_path_id += decl_id;
    }

    base_path_id += "@";
    base_path_id += class_id;
  }

  std::string vbo_symb_type_name = vbase_offset_type_prefix + base_class_id;
  const symbolt *vt_symb_type = ns.lookup(vbo_symb_type_name);
  assert(vt_symb_type);

  symbolt vbot_symb_var;
  vbot_symb_var.id = vbase_offset_type_prefix + base_path_id;
  vbot_symb_var.name = vbase_offset_type_prefix + base_path_id;
  vbot_symb_var.mode = mode;
  vbot_symb_var.module =
    get_modulename_from_path(type.location().file().as_string());
  vbot_symb_var.location = vt_symb_type->location;
  vbot_symb_var.type = symbol_typet(vt_symb_type->id);
  vbot_symb_var.lvalue = true;
  vbot_symb_var.static_lifetime = true;

  // add vtable variable symbols
  const struct_typet &vbot_type = to_struct_type(vt_symb_type->type);
  exprt values("struct", symbol_typet(vt_symb_type->id));
  for (const auto &compo : vbot_type.components())
  {
    exprt offset;
    if (cpp_base_offset::offset_to_base(compo.name(), type, offset, ns))
    {
      log_error(
        "Failed to calculate offset to virtual base {} for class {}",
        compo.name(),
        class_id);
      abort();
    }
    exprt other_offset = gen_zero(size_type());
    for (const auto &path_element : path)
    {
      exprt summand;
      const clang::CXXRecordDecl *sub_object =
        path_element.Base->getType()->getAsCXXRecordDecl();
      typet sub_object_enclosing_type;
      if (get_type(
            *path_element.Class->getTypeForDecl(),
            sub_object_enclosing_type,
            true))
      {
        log_error("Failed to get type for class {}", class_id);
        abort();
      }
      sub_object_enclosing_type = ns.follow(sub_object_enclosing_type);
      std::string decl_id, decl_name;
      get_decl_name(*sub_object, decl_name, decl_id);
      if (cpp_base_offset::offset_to_base(
            decl_name, sub_object_enclosing_type, summand, ns))
      {
        log_error(
          "Failed to calculate offset to virtual base {} for class {}",
          decl_name,
          class_id);
        abort();
      }
      other_offset = plus_exprt(other_offset, summand);
    }
    simplify(other_offset);

    exprt value = minus_exprt(offset, other_offset);
    value.type() = size_type();
    assert(value.type() == compo.type());
    values.operands().push_back(value);
  }
  vbot_symb_var.value = values;

  if (context.move(vbot_symb_var))
  {
    log_error(
      "Failed to add virtual base offset table variable symbol {} for "
      "class "
      "{}",
      vbot_symb_var.id,
      class_id);
    abort();
  }
}
