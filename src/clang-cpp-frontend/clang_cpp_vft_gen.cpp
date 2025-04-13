
#include "clang_cpp_vft_gen.h"

#include <clang-c-frontend/clang_c_convert.h>

#include <util/expr_util.h>
#include <util/message.h>

void clang_cpp_vft_gen::handle_vtable_and_vptr_generation(struct_typet &type)
{
  /*
  * 1. If this class has virtual methods,
  *  add a virtual table type symbol and virtual pointer. Then add a new
  *  entry in the vtable.
  */
  if (!type.get_bool("#has_vptr_component"))
    return;
  symbolt *vtable_type_symbol = check_vtable_type_symbol_existence(type);
  if (!vtable_type_symbol)
  {
    // first time we create the vtable type for this class
    vtable_type_symbol = add_vtable_type_symbol(type);
    assert(vtable_type_symbol);
  }
  add_vptr(type);

  /*
   * 2. add an entry in the existing virtual table type symbol
   */
  for (auto &comp : type.methods())
  {
    if (comp.get_bool("is_virtual") && !comp.get_bool("from_base"))
    {
      // this is a virtual method
      add_vtable_type_entry(type, comp, vtable_type_symbol);
    }
  }

  /*
   * Set up virtual function table(vft) variable symbols
   * Each vft is modelled as a struct of function pointers.
   */
  setup_vtable_struct_variables(type);
}

void clang_cpp_vft_gen::add_vtable_type_entry(
  const struct_typet &type,
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

  irep_idt vt_name = clang_cpp_vft_gen::vtable_type_prefix +
                     clang_cpp_vft_gen::tag_prefix + type.tag().as_string();
  std::string virtual_name = comp.name().as_string();
  struct_typet::componentt vt_entry;
  vt_entry.type() = pointer_typet(comp.type());
  vt_entry.set_name(vt_name.as_string() + "::" + virtual_name);
  vt_entry.set("base_name", comp.base_name());
  /*
   * `pretty_name` gets printed in symbol table:
   *    virtual_table::BLAH@tag-BLAH={ .<pretty_name>=&<virtual_method_base_class> };
   *    virtual_table::BLAH@tag-BLEH={ .<pretty_name>=&<thunk_to_overriding_method_in_derived_class> };
   *    virtual_table::BLEH@tag-BLEH={ .<pretty_name>=&<overriding_function_in_derived_class> };
   */
  vt_entry.set("pretty_name", comp.get("virtual_name"));
  vt_entry.set("virtual_name", comp.get("virtual_name"));
  vt_entry.set("access", "public");
  vt_entry.location() = comp.location();
  // add an entry to the virtual table
  assert(vtable_type_symbol);
  struct_typet &vtable_type = to_struct_type(vtable_type_symbol->type);
  vtable_type.components().push_back(vt_entry);
}

symbolt *
clang_cpp_vft_gen::check_vtable_type_symbol_existence(const struct_typet &type)
{
  irep_idt vt_name =
    clang_cpp_vft_gen::vtable_type_prefix + tag_prefix + type.tag().as_string();
  return context.find_symbol(vt_name);
}

symbolt *clang_cpp_vft_gen::add_vtable_type_symbol(struct_typet &type) const
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

  irep_idt vt_name =
    clang_cpp_vft_gen::vtable_type_prefix + tag_prefix + type.tag().as_string();

  const locationt &location = type.location();

  symbolt vt_type_symb;
  vt_type_symb.id = vt_name;
  vt_type_symb.name =
    clang_cpp_vft_gen::vtable_type_prefix + type.tag().as_string();
  vt_type_symb.mode = "C++";
  vt_type_symb.type = struct_typet();
  vt_type_symb.is_type = true;
  vt_type_symb.type.set("name", vt_type_symb.id);
  vt_type_symb.location = location;
  vt_type_symb.module =
    clang_c_convertert::get_modulename_from_path(location.file().as_string());

  if (context.move(vt_type_symb))
  {
    log_error(
      "Failed add vtable type symbol {} to symbol table", vt_type_symb.id);
    abort();
  }

  return context.find_symbol(vt_name);
}

void clang_cpp_vft_gen::add_vptr(struct_typet &type)
{
  /*
   * We model the virtual pointer as a `component` to the parent class' type.
   * This will be the vptr pointing to the vtable that contains the overriden functions.
   *
   * Vptr has the name in the form of `tag-BLAH@vtable_pointer`, where BLAH is the class name.
   */

  irep_idt vt_name = clang_cpp_vft_gen::vtable_type_prefix +
                     clang_cpp_vft_gen::tag_prefix + type.tag().as_string();
  // add a virtual-table pointer
  struct_typet::componentt component;
  component.type() = pointer_typet(symbol_typet(vt_name));
  component.set_name(
    clang_cpp_vft_gen::tag_prefix + type.tag().as_string() +
    "::" + clang_cpp_vft_gen::vtable_ptr_suffix);
  component.base_name(clang_cpp_vft_gen::vtable_ptr_suffix);
  component.pretty_name(
    type.tag().as_string() + clang_cpp_vft_gen::vtable_ptr_suffix);
  component.set("is_vtptr", true);
  component.set("access", "public");
  // add to the class' type
  type.components().push_back(component);
}

void clang_cpp_vft_gen::setup_vtable_struct_variables(const struct_typet &type)
{
  /*
   * Recall that we model the virtual function table (VFT) as
   * a struct of function pointers each pointing to
   * the targeting function, where this targeting function can be a
   * virtual, thunk or the actual overriding function.
   *
   * To add the VFT struct variables, we follow the steps below:
   *  1. Build an intermediate VFT value map from the class' type
   *  2. Create the VFT symbols based on this map. These symbols are
   *  of the VFT struct type we created when getting virtual methods.
   */

  switch_table vtable_value_map;

  build_vtable_map(type, vtable_value_map);

  add_vtable_variable_symbols(type, vtable_value_map);
}

void clang_cpp_vft_gen::build_vtable_map(
  const struct_typet &struct_type,
  switch_table &vtable_value_map)
{
  /*
   * Build a vtable map from the class type
   * This is the virtual function table for this class.
   * This table will be used to create the vtable variable symbols.
   */

  for (const auto &method : struct_type.methods())
  {
    if (!method.get_bool("is_virtual"))
      continue;

    const code_typet &code_type = to_code_type(method.type());
    assert(code_type.arguments().size() >= 1); // because of `this` param

    const pointer_typet &pointer_type =
      static_cast<const pointer_typet &>(code_type.arguments()[0].type());

    irep_idt class_id = pointer_type.subtype().identifier();

    std::map<irep_idt, exprt> &value_map =
      vtable_value_map[class_id]; // switch_map = switch_table
    exprt e = symbol_exprt(method.get_name(), code_type);

    dstring virtual_name = method.get("virtual_name");
    assert(!virtual_name.empty());
    if (method.get_bool("is_pure_virtual"))
    {
      pointer_typet pointer_type(code_type);
      e = gen_zero(pointer_type);
      assert(e.is_not_nil());
      value_map[virtual_name] = e;
    }
    else
    {
      address_of_exprt address(e);
      value_map[virtual_name] = address;
    }
  }
}

void clang_cpp_vft_gen::add_vtable_variable_symbols(
  const struct_typet &type,
  const switch_table &vtable_value_map)
{
  const std::string class_id =
    clang_cpp_vft_gen::tag_prefix + type.tag().as_string();
  /*
   * Now we got the vtable map for the class type representing
   * the function switch relations.
   * Let's use this map to create the vtable struct variables
   * and add them to the symbol table.
   */

  for (const auto &vft_switch_kv_pair : vtable_value_map)
  {
    const function_switch &switch_map = vft_switch_kv_pair.second;

    // To create the vtable variable symbol we need to get the corresponding type
    const symbolt *late_cast_symb = ns.lookup(vft_switch_kv_pair.first);
    assert(late_cast_symb);
    const symbolt *vt_symb_type =
      ns.lookup("virtual_table::" + late_cast_symb->id.as_string());
    assert(vt_symb_type);

    symbolt vt_symb_var;
    vt_symb_var.id = vt_symb_type->id.as_string() + "@" + class_id;
    vt_symb_var.name = vt_symb_type->name.as_string() + "@" + class_id;
    vt_symb_var.mode = "C++";
    vt_symb_var.module = clang_c_convertert::get_modulename_from_path(
      type.location().file().as_string());
    vt_symb_var.location = vt_symb_type->location;
    vt_symb_var.type = symbol_typet(vt_symb_type->id);
    vt_symb_var.lvalue = true;
    vt_symb_var.static_lifetime = true;

    // add vtable variable symbols
    const struct_typet &vt_type = to_struct_type(vt_symb_type->type);
    exprt values("struct", symbol_typet(vt_symb_type->id));
    for (const auto &compo : vt_type.components())
    {
      std::map<irep_idt, exprt>::const_iterator cit2 =
        switch_map.find(compo.get("virtual_name").as_string());
      assert(cit2 != switch_map.end());
      const exprt &value = cit2->second;
      assert(value.type().id() == compo.type().id());
      values.operands().push_back(value);
    }
    vt_symb_var.value = values;

    if (context.move(vt_symb_var))
    {
      log_error(
        "Failed to add vtable variable symbol {} for class {}",
        vt_symb_var.id,
        class_id);
      abort();
    }
  }
}