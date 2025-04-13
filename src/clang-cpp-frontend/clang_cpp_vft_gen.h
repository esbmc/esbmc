
#ifndef CLANG_CPP_VFT_GEN_H
#define CLANG_CPP_VFT_GEN_H
#include <map>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/std_types.h>

class clang_cpp_vft_gen
{
public:
  explicit clang_cpp_vft_gen(contextt &_context)
    : context(_context), ns(context)
  {
  }

  void handle_vtable_and_vptr_generation(struct_typet &type);
  static inline std::string vtable_type_prefix = "virtual_table::";
  static inline std::string vtable_ptr_suffix = "@vtable_pointer";

  /*
   * Check the existence of virtual table type symbol.
   * If it exists, return its pointer. Otherwise, return nullptr.
   */
  symbolt *check_vtable_type_symbol_existence(const struct_typet &type) const;
  /*
   * Add virtual table(vtable) type symbol.
   * This is added as a type symbol in the symbol table.
   *
   * This is done the first time when we encounter a virtual method in a class
   */
  symbolt *add_vtable_type_symbol(struct_typet &type) const;

protected:
  contextt &context;
  namespacet ns;
  static inline std::string tag_prefix = "tag-";
  using function_switch = std::map<irep_idt, exprt>;
  using switch_table = std::map<irep_idt, function_switch>;

  /*
   * Add virtual pointer(vptr).
   * Vptr is NOT a symbol but rather simply added as a component to the class' type.
   *
   * This is done the first time we encounter a virtual method in a class
   */
  void add_vptr(struct_typet &type);
  /*
   * Add an entry to the virtual table type
   *
   * This is done when NOT the first time we encounter a virtual method in a class
   * in which case we just want to add a new entry to the virtual table type
   */
  void add_vtable_type_entry(
    const struct_typet &type,
    struct_typet::componentt &comp,
    symbolt *vtable_type_symbol);
  symbolt *check_vtable_type_symbol_existence(const struct_typet &type);
  /*
   * Recall that we model the virtual function table as struct of function pointers.
   * This function adds the symbols for these struct variables.
   *
   * Params:
   *  - type: ESBMC IR representing the type the class/struct we are currently dealing with
   */
  void setup_vtable_struct_variables(const struct_typet &type);
  /*
   * This function builds the vtable value map -
   * a map representing the function switch table
   * with each key-value pair in the form of:
   *  Class X : {VirtualName Y : FunctionID}
   *
   * where X represents the name of a virtual/thunk/overriding function and function ID represents the
   * actual function we are calling when calling the virtual/thunk/overriding function
   * via a Class X* pointer, something like:
   *   xptr->Y()
   *
   * Params:
   *  - struct_type: ESBMC IR representing the type of the class/struct we are currently dealing with
   *  - vtable_value_map: representing the vtable value maps for this class/struct we are currently dealing with
   */
  void build_vtable_map(
    const struct_typet &struct_type,
    switch_table &vtable_value_map);
  /*
   * Create the vtable variable symbols and add them to the symbol table.
   * Each vtable variable represents the actual function switch table, which
   * is modelled as a struct of function pointers, e.g.:
   *  Vtable tag.Base@Base =
   *    {
   *      .do_something = &TagBase::do_someting();
   *    };
   *
   * Params:
   *  - struct_type: ESBMC IR representing the type the class/struct we are currently dealing with
   *  - vtable_value_map: representing the vtable value maps for this class/struct we are currently dealing with
   */
  void add_vtable_variable_symbols(
    const struct_typet &struct_type,
    const switch_table &vtable_value_map);
};

#endif //CLANG_CPP_VFT_GEN_H
