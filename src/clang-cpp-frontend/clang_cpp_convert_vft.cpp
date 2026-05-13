/*
 * This file contains functions to generate virtual function table (VFT):
 *  - travere virtual methods
 *  - generate VFT type symbol
 *  - generate VFT variable symbols
 *  - generate thunk functions for overriding methods
 */
#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Basic/Version.inc>
#include <clang/AST/Attr.h>
#include <clang/AST/CXXInheritance.h>
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
#include <clang/AST/RecordLayout.h>
#include <llvm/Support/raw_os_ostream.h>
CC_DIAGNOSTIC_POP()

#include <clang-c-frontend/typecast.h>
#include <clang-cpp-frontend/clang_cpp_convert.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/std_code.h>
#include <util/std_expr.h>

#include <functional>
#include <optional>

bool clang_cpp_convertert::get_struct_class_virtual_methods(
  const clang::CXXRecordDecl &cxxrd,
  struct_typet &type)
{
  // Register cxxrd against any inherited vptr-class up front so that an
  // inline body containing dynamic_cast<cxxrd&>(base_ref) — converted by
  // the loop below — can match its own runtime type.
  pre_register_inherited_vtables(cxxrd);

  for (const auto &md : cxxrd.methods())
  {
    if (!md->isVirtual())
      continue;

    /*
     * 1. convert this virtual method and add them to class symbol type
     */
    struct_typet::componentt comp;
    if (get_decl(*md, comp))
      return true;

    // additional annotations for virtual/overriding methods
    if (annotate_virtual_overriding_methods(*md, comp))
      return true;
    type.methods().push_back(comp);

    /*
     * 2. If this is the first time we see a virtual method in this class,
     *  add virtual table type symbol and virtual pointer. Then add a new
     *  entry in the vtable.
     */
    symbolt *vtable_type_symbol = check_vtable_type_symbol_existence(type);
    if (!vtable_type_symbol)
    {
      // first time we create the vtable type for this class
      vtable_type_symbol = add_vtable_type_symbol(comp, type);
      if (vtable_type_symbol == nullptr)
        return true;

      add_vptr(type);
    }

    /*
     * 3. add an entry in the existing virtual table type symbol
     */
    add_vtable_type_entry(type, comp, vtable_type_symbol);

    /*
     * 4. deal with overriding method
     */
    if (md->begin_overridden_methods() != md->end_overridden_methods())
    {
      /*
       * In a multi-inheritance case(e.g. diamond problem)
       * a method might overrides multiple base methods in multiple levels.
       * so we need to create multiple thunk functions for each overriden
       * method in each level.
       */
      overriden_map cxxmethods_overriden;
      get_overriden_methods(*md, cxxmethods_overriden);

      for (const auto &overriden_md_entry : cxxmethods_overriden)
        add_thunk_method(cxxrd, overriden_md_entry.second, comp, type);
    }
  }

  /*
   * Set up virtual function table(vft) variable symbols
   * Each vft is modelled as a struct of function pointers.
   */
  setup_vtable_struct_variables(cxxrd, type);

  return false;
}

static bool is_pure_virtual(const clang::CXXMethodDecl &md)
{
#if CLANG_VERSION_MAJOR < 18
  return md.isPure();
#else
  return md.isPureVirtual();
#endif
}

/**
 * @brief Returns the ultimate overridden method for a given CXXMethodDecl.
 *
 * This function traverses the overridden methods of the provided CXXMethodDecl
 * and returns the **unique** ultimate overridden method.
 *
 * @param method The CXXMethodDecl to find the ultimate overridden method for.
 * @return The unique ultimate overridden CXXMethodDecl.
 */
static const clang::CXXMethodDecl *
get_ultimate_overridden_method(const clang::CXXMethodDecl *method)
{
  const clang::CXXMethodDecl *current_method = method;
  while (current_method->size_overridden_methods() == 1)
  {
    current_method = *current_method->overridden_methods().begin();
  }
  return current_method;
}

bool clang_cpp_convertert::annotate_virtual_overriding_methods(
  const clang::CXXMethodDecl &md,
  struct_typet::componentt &comp)
{
  const clang::CXXMethodDecl *ultimate_overridden_method =
    get_ultimate_overridden_method(&md);
  std::string overridden_method_id, overridden_method_name;
  get_decl_name(
    *ultimate_overridden_method, overridden_method_name, overridden_method_id);
  std::string virtual_name = overridden_method_id;

  comp.set("is_virtual", true);
  comp.set("virtual_name", virtual_name);

  if (is_pure_virtual(md))
    comp.set("is_pure_virtual", true);

  return false;
}

symbolt *clang_cpp_convertert::check_vtable_type_symbol_existence(
  const struct_typet &type)
{
  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();
  return context.find_symbol(vt_name);
}

symbolt *clang_cpp_convertert::add_vtable_type_symbol(
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

  if (context.move(vt_type_symb))
  {
    log_error(
      "Failed add vtable type symbol {} to symbol table", vt_type_symb.id);
    abort();
  }

  return context.find_symbol(vt_name);
}

void clang_cpp_convertert::add_vptr(struct_typet &type)
{
  /*
   * We model the virtual pointer as a `component` to the parent class' type.
   * This will be the vptr pointing to the vtable that contains the overriden functions.
   *
   * Vptr has the name in the form of `tag-BLAH@vtable_pointer`, where BLAH is the class name.
   */

  irep_idt vt_name = vtable_type_prefix + tag_prefix + type.tag().as_string();
  // add a virtual-table pointer
  struct_typet::componentt component;
  component.type() = pointer_typet(symbol_typet(vt_name));
  component.set_name(
    tag_prefix + type.tag().as_string() + "::" + vtable_ptr_suffix);
  component.base_name(vtable_ptr_suffix);
  component.pretty_name(type.tag().as_string() + vtable_ptr_suffix);
  component.set("is_vtptr", true);
  component.set("access", "public");
  // add to the class' type
  type.components().push_back(component);

  has_vptr_component = true;
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

void clang_cpp_convertert::add_thunk_method(
  const clang::CXXRecordDecl &derived_rd,
  const clang::CXXMethodDecl &md,
  const struct_typet::componentt &component,
  struct_typet &type)
{
  /*
   * Add a thunk function for an overriding method.
   * This thunk function will be added as a symbol in the symbol table,
   * and considered a `component` to the derived class' type.
   * This thunk function will be used to set up the derived class' vtable
   * to override the base method, e.g.
   *
   * Suppose Penguin derives Bird, we have the following vtables for Penguin:
   *  virtual_table::Bird@Penguin =
   *    {
   *      .do_it() = &thunk::c:@S@Penguin@F@do_something#::tag-Bird; // this is the thunk redirecting call to the overriding function
   *    };
   *
   *  virtual_table::Penguin@Penguin =
   *    {
   *      .do_it() = &c:@S@Penguin@F@do_something#::do_it(); // this is the overriding function
   *    };
   *
   *  The thunk function's symbol id is of the form - "thunk::c:@S@Penguin@F@do_something#::tag-Bird"
   *  meaning "a thunk to Penguin's overriding function `do_something` taking a `this` parameter of the type Bird*"
   */

  /*
   * For this thunk method, we need to add:
   *  1. its symbol in the symbol table
   *  2. its arguments in the symbol table
   *  3. its body
   *
   *  also need to add this thunk method to the list of components of the derived class' type
   */

  std::string base_class_id, base_class_name;
  get_decl_name(*md.getParent(), base_class_name, base_class_id);

  // Compute the byte offset of the base class sub-object within the derived
  // class. For non-first base classes in multiple inheritance this is non-zero,
  // and the thunk must subtract it from its Base* this to recover Derived*.
  // TODO: This loop only searches direct bases of derived_rd. If base_rd is
  // an indirect base (e.g. Derived : Middle, Middle : B8), base_offset will
  // incorrectly remain 0. A complete fix requires summing offsets along the
  // full inheritance path using CXXBasePaths::isDerivedFrom or a recursive
  // walk — to be addressed as a follow-up.
  uint64_t base_offset = 0;
  const clang::CXXRecordDecl *base_rd = md.getParent();
  const clang::ASTRecordLayout &layout =
    ASTContext->getASTRecordLayout(&derived_rd);
  for (const auto &base_spec : derived_rd.bases())
  {
    const clang::CXXRecordDecl *spec_rd =
      base_spec.getType()->getAsCXXRecordDecl();
    if (spec_rd == base_rd && !base_spec.isVirtual())
    {
      base_offset = layout.getBaseClassOffset(base_rd).getQuantity();
      break;
    }
  }

  // Create the thunk method symbol
  symbolt thunk_func_symb;
  thunk_func_symb.id =
    base_class_id + "::" + thunk_prefix + component.get_name().as_string();
  thunk_func_symb.name = component.base_name();
  thunk_func_symb.mode = mode;
  thunk_func_symb.location = component.location();
  thunk_func_symb.type = component.type();
  thunk_func_symb.module =
    get_modulename_from_path(component.location().file().as_string());

  // make thunk function names intuitive
  set_thunk_name(thunk_func_symb, base_class_id);

  // update the type of `this` argument in thunk
  update_thunk_this_type(thunk_func_symb.type, base_class_id);

  // add symbols for arguments of this thunk function
  add_thunk_method_arguments(thunk_func_symb);

  // add thunk function body
  add_thunk_method_body(thunk_func_symb, component, base_offset);

  // add thunk function symbol to the symbol table
  symbolt &added_thunk_symbol =
    *context.move_symbol_to_context(thunk_func_symb);

  // add thunk function as a `method` in the derived class' type
  add_thunk_component_to_type(added_thunk_symbol, type, component);
}

void clang_cpp_convertert::set_thunk_name(
  symbolt &thunk_func_symb,
  const std::string &base_class_id)
{
  /*
   * set an intuitive name to thunk function in the form of:
   *  thunk_to::<overriding_func_name>::tag-Base
   * where `Base` represents the base function this overriding function
   * applies to.
   */

  irep_idt thunk_bn = base_class_id + "::" + thunk_prefix +
                      "to::" + thunk_func_symb.name.as_string();
  thunk_func_symb.name = thunk_bn;
}

void clang_cpp_convertert::update_thunk_this_type(
  typet &thunk_symbol_type,
  const std::string &base_class_id)
{
  /*
   * update the type of `this` argument in thunk function:
   * Before:
   *  * type: pointer
        * subtype: symbol
          * identifier: tag-Derived
     After:
   *  * type: pointer
        * subtype: symbol
          * identifier: tag-Base
    */

  code_typet &code_type = to_code_type(thunk_symbol_type);
  code_typet::argumentt &this_arg = code_type.arguments().front();
  this_arg.type().subtype().set("identifier", base_class_id);
}

void clang_cpp_convertert::add_thunk_method_arguments(symbolt &thunk_func_symb)
{
  /*
   * Loop through the arguments of the thunk methods,
   * and add symbol for each argument. We need to
   * update the identifier field in each argument to indicate it "belongs" to the thunk function.
   *
   * Each argument symbol's id is of the form - "<thunk_func_symbol_ID>::<argument_base_name>"
   */

  code_typet &code_type = to_code_type(thunk_func_symb.type);
  code_typet::argumentst &args = code_type.arguments();
  for (unsigned i = 0; i < args.size(); i++)
  {
    code_typet::argumentt &arg = args[i];
    irep_idt base_name = arg.get_base_name();

    symbolt arg_symb;
    arg_symb.id = thunk_func_symb.id.as_string() + "::" + base_name.as_string();
    arg_symb.name = base_name;
    arg_symb.mode = mode;
    arg_symb.location = thunk_func_symb.location;
    arg_symb.type = arg.type();

    // Change argument identifier field to thunk function
    arg.set("#identifier", arg_symb.id);

    // add the argument to the symbol table
    symbolt *tmp_symbol;
    if (context.move(arg_symb, tmp_symbol))
    {
      log_error(
        "Failed to add arg symbol `{}' for thunk function `{}'.\n"
        "`{}' already exists",
        arg_symb.id,
        thunk_func_symb.id,
        tmp_symbol->id);
      abort();
    }
  }
}

void clang_cpp_convertert::add_thunk_method_body(
  symbolt &thunk_func_symb,
  const struct_typet::componentt &component,
  uint64_t base_offset)
{
  code_typet &code_type = to_code_type(thunk_func_symb.type);
  code_typet::argumentst &args = code_type.arguments();

  // Build the adjusted 'this' to pass to the overriding method.
  // The thunk receives a Base*, but the overriding method expects Derived*.
  // For non-first base classes, subtract the byte offset of the Base
  // sub-object within Derived so that the overriding method's 'this' points
  // to the start of the Derived object, not the Base sub-object.
  typet derived_ptr_type = to_code_type(component.type()).arguments()[0].type();
  exprt base_this =
    symbol_expr(*namespacet(context).lookup(args[0].cmt_identifier()));

  exprt adjusted_this;
  if (base_offset > 0)
  {
    typet char_ptr = pointer_typet(char_type());
    typecast_exprt to_char(base_this, char_ptr);
    minus_exprt sub(to_char, from_integer(base_offset, index_type()));
    sub.type() = char_ptr;
    adjusted_this = typecast_exprt(sub, derived_ptr_type);
  }
  else
    adjusted_this = typecast_exprt(base_this, derived_ptr_type);

  if (
    code_type.return_type().id() != "empty" &&
    code_type.return_type().id() != "destructor")
    add_thunk_method_body_return(thunk_func_symb, component, adjusted_this);
  else
    add_thunk_method_body_no_return(thunk_func_symb, component, adjusted_this);
}

void clang_cpp_convertert::add_thunk_method_body_return(
  symbolt &thunk_func_symb,
  const struct_typet::componentt &component,
  const exprt &late_cast_this)
{
  /*
   * Add thunk function with return value, something like:
   * thunk_to_do_something(this): // `this` of the type `Base*`
   *  int return_value;
   *  FUNCTION_CALL: return_value = do_something((Derived*)this);
   *  RETURN: return_value;
   * END_FUNCTION
   */
  code_typet::argumentst &args = to_code_type(thunk_func_symb.type).arguments();

  side_effect_expr_function_callt expr_call;
  expr_call.function() = symbol_exprt(component.get_name(), component.type());
  expr_call.type() = to_code_type(component.type()).return_type();
  expr_call.arguments().reserve(args.size());
  // `this` parameter is always the first one to add
  expr_call.arguments().push_back(late_cast_this);

  for (unsigned i = 1; i < args.size(); i++)
  {
    expr_call.arguments().push_back(
      symbol_expr(*namespacet(context).lookup(args[i].cmt_identifier())));
  }

  code_returnt code_return;
  code_return.return_value() = expr_call;

  thunk_func_symb.value = code_return;
}

void clang_cpp_convertert::add_thunk_method_body_no_return(
  symbolt &thunk_func_symb,
  const struct_typet::componentt &component,
  const exprt &late_cast_this)
{
  /*
   * Add thunk function without return value, something like:
   * thunk_to_do_something(this): // `this` of the type `Base*`
   *  FUNCTION_CALL: do_something((Derived*)this);
   * END_FUNCTION
   */
  code_typet::argumentst &args = to_code_type(thunk_func_symb.type).arguments();

  code_function_callt code_func;
  code_func.function() = symbol_exprt(component.get_name(), component.type());
  code_func.arguments().reserve(args.size());
  // `this` parameter is always the first one to add
  code_func.arguments().push_back(late_cast_this);

  for (unsigned i = 1; i < args.size(); i++)
  {
    code_func.arguments().push_back(
      symbol_expr(*namespacet(context).lookup(args[i].cmt_identifier())));
  }

  thunk_func_symb.value = code_func;
}

void clang_cpp_convertert::add_thunk_component_to_type(
  const symbolt &thunk_func_symb,
  struct_typet &type,
  const struct_typet::componentt &comp)
{
  struct_typet::componentt new_compo = comp;
  new_compo.type() = thunk_func_symb.type;
  new_compo.set_name(thunk_func_symb.id);
  type.methods().push_back(new_compo);
}

void clang_cpp_convertert::setup_vtable_struct_variables(
  const clang::CXXRecordDecl &cxxrd,
  const struct_typet &type)
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

  add_vtable_variable_symbols(cxxrd, type, vtable_value_map);
}

void clang_cpp_convertert::build_vtable_map(
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

    irep_idt virtual_name = method.get("virtual_name");
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

void clang_cpp_convertert::add_vtable_variable_symbols(
  const clang::CXXRecordDecl &cxxrd,
  const struct_typet &type,
  const switch_table &vtable_value_map)
{
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

    // This is the class we are currently dealing with
    std::string class_id, class_name;
    get_decl_name(cxxrd, class_name, class_id);

    symbolt vt_symb_var;
    vt_symb_var.id = vt_symb_type->id.as_string() + "@" + class_id;
    vt_symb_var.name = vt_symb_type->name.as_string() + "@" + class_id;
    vt_symb_var.mode = mode;
    vt_symb_var.module =
      get_modulename_from_path(type.location().file().as_string());
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

    // Record (vptr-class V → concrete class D) so build_dynamic_cast can
    // enumerate candidate D's by direct lookup instead of walking the TU.
    // Skip abstract classes: no object of an abstract type can exist at
    // runtime, so they can never be the answer to dynamic_cast.
    if (!cxxrd.isAbstract())
      vtable_classes_per_vptr_[late_cast_symb->id].insert(&cxxrd);
  }
}

void clang_cpp_convertert::get_overriden_methods(
  const clang::CXXMethodDecl &md,
  overriden_map &map)
{
  /*
   * This function gets all the overriden methods to which we need to create a thunk
   */
  for (const auto &md_overriden : md.overridden_methods())
  {
    if (
      md_overriden->begin_overridden_methods() !=
      md_overriden->end_overridden_methods())
      get_overriden_methods(*md_overriden, map);

    // get the id for this overriden method
    std::string method_id, method_name;
    get_decl_name(*md_overriden, method_name, method_id);

    // avoid adding the same overriden method, e.g. in case of diamond problem
    if (map.find(method_id) != map.end())
      continue;

    auto status = map.insert({method_id, *md_overriden});
    (void)status;
    assert(status.second);
  }
}

// Compute the byte offset of subobject S inside D. Returns std::nullopt if
// S isn't reachable from D via non-virtual inheritance (virtual base on the
// path, or S not a base of D at all): callers fall back to a structural
// typecast in that case.
static std::optional<uint64_t> offset_of_subobject(
  const clang::ASTContext &ctx,
  const clang::CXXRecordDecl *D,
  const clang::CXXRecordDecl *S)
{
  if (D == S)
    return 0;

  clang::CXXBasePaths paths(
    /*FindAmbiguities=*/false,
    /*RecordPaths=*/true,
    /*DetectVirtual=*/true);
  if (!D->isDerivedFrom(S, paths))
    return std::nullopt;
  if (paths.getDetectedVirtual() != nullptr)
    return std::nullopt;

  // Sum base-class offsets along the first recorded path. Multiple paths
  // can exist under repeated non-virtual inheritance, but they refer to
  // distinct subobjects; picking the first matches the behaviour of the
  // previous primary-base walk.
  uint64_t offset = 0;
  const clang::CXXRecordDecl *cur = D;
  for (const clang::CXXBasePathElement &elem : paths.front())
  {
    const clang::CXXRecordDecl *base =
      elem.Base->getType()->getAsCXXRecordDecl();
    offset +=
      ctx.getASTRecordLayout(cur).getBaseClassOffset(base).getQuantity();
    cur = base;
  }
  return offset;
}

void clang_cpp_convertert::pre_register_inherited_vtables(
  const clang::CXXRecordDecl &cxxrd)
{
  // add_vtable_variable_symbols populates vtable_classes_per_vptr_ at the
  // tail of get_struct_class_virtual_methods, after method bodies have
  // been converted. That's too late for an inline body that does
  // dynamic_cast<T&>(src) where T is the enclosing class itself
  // (e.g. IntCoord::assign casting `const Coord&` to `const IntCoord&`).
  // Pre-seed the map for every non-virtual ancestor whose vtable type
  // already exists, so build_dynamic_cast's lookup hits regardless of how
  // far up the chain it resolves the vptr-class to.
  //
  // Abstract: skipped because no object of an abstract type exists at
  // runtime, so cxxrd can never be the answer to dynamic_cast.
  if (cxxrd.isAbstract())
    return;

  std::function<void(const clang::CXXRecordDecl *)> walk =
    [&](const clang::CXXRecordDecl *cur) {
      for (const auto &spec : cur->bases())
      {
        if (spec.isVirtual())
          continue;
        const auto *base = spec.getType()->getAsCXXRecordDecl();
        if (!base)
          continue;
        std::string base_id, base_name;
        get_decl_name(*base, base_name, base_id);
        if (ns.lookup(vtable_type_prefix + base_id))
          vtable_classes_per_vptr_[base_id].insert(&cxxrd);
        walk(base);
      }
    };
  walk(&cxxrd);
}

bool clang_cpp_convertert::build_dynamic_cast(
  const clang::CXXDynamicCastExpr &cast,
  exprt &new_expr)
{
  // Lower dynamic_cast into a guarded ITE that reads src's vptr and
  // compares it against vtable variable addresses to identify the
  // runtime type. For dynamic_cast<T*>:
  //   src == NULL ? (T*)NULL
  //               : (src->vptr == &vt-D1 || src->vptr == &vt-D2 || ...)
  //                   ? (T*)src
  //                   : (T*)NULL
  // For dynamic_cast<void*> the result is a pointer to the start of the
  // most-derived object, so each matching D contributes its own arm
  // adjusting src by the offset of S inside D.
  // For dynamic_cast<T&> the result is wrapped in a statement_expression
  // that throws std::bad_cast when no D matches, so the catch (bad_cast&)
  // arm of a try/catch can fire.

  exprt sub;
  if (get_expr(*cast.getSubExpr(), sub))
    return true;
  typet target_type;
  if (get_type(cast.getType(), target_type))
    return true;

  auto fallback = [&]() {
    gen_typecast(ns, sub, target_type);
    new_expr = sub;
    return false;
  };

  // Reference dynamic_cast: Clang strips the reference from getType() but
  // the result expression is still an lvalue (or xvalue for rvalue
  // references), so the value kind tells us the source form.
  const bool is_reference = !cast.isPRValue();
  if (!is_reference && !cast.getType()->isPointerType())
    return fallback();

  // Normalise the source side: for the pointer form sub is already a
  // pointer to S; for the reference form sub is an lvalue of S, which we
  // wrap in address_of so the rest of the helper can treat both shapes
  // uniformly.
  clang::QualType src_clang_type = cast.getSubExpr()->getType();
  clang::QualType src_pointee_qt;
  exprt src_pointer;
  if (is_reference)
  {
    src_pointee_qt = src_clang_type->isReferenceType()
                       ? src_clang_type->getPointeeType()
                       : src_clang_type;
    if (sub.type().is_pointer())
      src_pointer = sub;
    else
      src_pointer = address_of_exprt(sub);
  }
  else
  {
    if (!src_clang_type->isPointerType())
      return fallback();
    src_pointee_qt = src_clang_type->getPointeeType();
    src_pointer = sub;
  }

  // For the pointer form, getType() is T* and the pointee is T. For the
  // reference form Clang has already stripped the &, so getType() *is* T.
  clang::QualType tgt_pointee_qt =
    is_reference ? cast.getType() : cast.getType()->getPointeeType();

  const clang::CXXRecordDecl *S = src_pointee_qt->getAsCXXRecordDecl();
  if (!S || !S->hasDefinition())
    return fallback();

  const bool to_void = !is_reference && tgt_pointee_qt->isVoidType();
  const clang::CXXRecordDecl *T = nullptr;
  if (!to_void)
  {
    T = tgt_pointee_qt->getAsCXXRecordDecl();
    if (!T || !T->hasDefinition())
      return fallback();
    // Static upcast / identity: the layout already matches, so the legacy
    // structural typecast is exact and no runtime check is needed.
    if (S == T || S->isDerivedFrom(T))
      return fallback();
  }

  // Walk up S's primary inheritance chain to find the class V that owns
  // the vptr (the most-derived class on the chain with a
  // virtual_table::tag-V type symbol). A virtual base on the path
  // would require runtime offset adjustment via a virtual-base table
  // we don't model, so refuse the cast loudly rather than silently
  // typecast and risk confidently-wrong verifications.
  std::string vptr_class_id;
  for (const clang::CXXRecordDecl *cur = S; cur != nullptr;)
  {
    std::string id, name;
    get_decl_name(*cur, name, id);
    if (ns.lookup(vtable_type_prefix + id))
    {
      vptr_class_id = id;
      break;
    }
    const clang::CXXRecordDecl *next = nullptr;
    for (const auto &spec : cur->bases())
    {
      if (spec.isVirtual())
      {
        log_error(
          "dynamic_cast through virtual inheritance is not supported "
          "(class `{}` has a virtual base)",
          cur->getNameAsString());
        abort();
      }
      if (const auto *bd = spec.getType()->getAsCXXRecordDecl())
      {
        next = bd;
        break;
      }
    }
    cur = next;
  }
  if (vptr_class_id.empty())
    return fallback();

  // Enumerate concrete D's via the side table populated during vtable
  // registration: every entry is a class with a real vtable variable for
  // this vptr V, so no per-cast TU walk and no symbol-table probe.
  // For T*, the matching set is additionally constrained to D being
  // at-or-below T; for void* every such D contributes (its result is a
  // pointer adjusted to D's start). Each entry is (vtable-variable
  // address, per-D result expression).
  pointer_typet vptr_type(symbol_typet(vtable_type_prefix + vptr_class_id));
  std::vector<std::pair<exprt, exprt>> arms;
  exprt typed_null = gen_zero(target_type);

  auto vptr_it = vtable_classes_per_vptr_.find(vptr_class_id);
  if (vptr_it != vtable_classes_per_vptr_.end())
  {
    for (const clang::CXXRecordDecl *D : vptr_it->second)
    {
      if (D != S && !D->isDerivedFrom(S))
        continue;
      if (!to_void && !(D == T || D->isDerivedFrom(T)))
        continue;

      std::string D_id, D_name;
      get_decl_name(*D, D_name, D_id);
      const std::string vt_var_id =
        vtable_type_prefix + vptr_class_id + "@" + D_id;
      exprt vt_addr =
        address_of_exprt(symbol_exprt(vt_var_id, vptr_type.subtype()));

      exprt result;
      if (to_void)
      {
        // (void*)((char*)src - off(S inside D))
        auto off = offset_of_subobject(*ASTContext, D, S);
        if (!off)
          continue;
        typet char_ptr = pointer_typet(char_type());
        exprt adj = src_pointer;
        gen_typecast(ns, adj, char_ptr);
        if (*off > 0)
        {
          adj = minus_exprt(adj, from_integer(*off, index_type()));
          adj.type() = char_ptr;
        }
        gen_typecast(ns, adj, target_type);
        result = adj;
      }
      else
      {
        // For T* the result must point to the T sub-object inside D:
        //   result = src + (off(T inside D) - off(S inside D))
        // When both offsets are zero (single inheritance with S and T
        // at the start of D) the structural typecast is exact.
        // Otherwise a per-D byte adjustment is required, which we
        // don't compute yet — refuse the cast instead of silently
        // emitting a pointer into the wrong sub-object.
        auto off_S = offset_of_subobject(*ASTContext, D, S);
        auto off_T = offset_of_subobject(*ASTContext, D, T);
        if (!off_S || !off_T)
        {
          log_error(
            "dynamic_cast: virtual base between runtime type `{}` and "
            "source/target is not supported",
            D->getNameAsString());
          abort();
        }
        if (*off_S != 0 || *off_T != 0)
        {
          log_error(
            "dynamic_cast: multiple inheritance with non-zero base "
            "offset in `{}` is not supported",
            D->getNameAsString());
          abort();
        }
        exprt adj = src_pointer;
        gen_typecast(ns, adj, target_type);
        result = adj;
      }
      arms.push_back({vt_addr, result});
    }
  }

  // src->vptr — needed by every form below that consults the runtime type.
  // Built once and copied into each arm; the IR is value-typed so this is
  // safe (no aliasing across exprts).
  exprt vptr_read;
  {
    exprt src_deref = dereference_exprt(src_pointer, src_pointer.type());
    vptr_read = member_exprt(
      src_deref, vptr_class_id + "::" + vtable_ptr_suffix, vptr_type);
  }

  // OR-chain: vptr == arm0 || vptr == arm1 || ... — used by the reference
  // form and the T* pointer form. Precondition: arms not empty.
  auto vptr_match_any = [&]() -> exprt {
    exprt match = equality_exprt(vptr_read, arms.front().first);
    for (size_t i = 1; i < arms.size(); ++i)
      match = or_exprt(match, equality_exprt(vptr_read, arms[i].first));
    return match;
  };

  // Reference form: if the vptr check fails, call __ESBMC_throw_bad_cast()
  // which symex resolves to a std::bad_cast throw at verification time.
  // This decouples the frontend from <typeinfo> inclusion order entirely.
  if (is_reference)
  {
    // Clang strips the reference from cast.getType(), so target_type is
    // the un-referenced struct T. Reconstruct the IR-level reference
    // type (pointer-to-T flagged as a reference) for the cast result.
    typet ref_type = pointer_typet(target_type);
    ref_type.set("#reference", true);

    exprt cast_value = src_pointer;
    gen_typecast(ns, cast_value, ref_type);

    exprt match = arms.empty() ? exprt(false_exprt()) : vptr_match_any();

    code_typet throw_func_type;
    throw_func_type.return_type() = empty_typet();
    symbol_exprt throw_func("c:@F@__ESBMC_throw_bad_cast", throw_func_type);

    code_function_callt throw_call;
    throw_call.function() = throw_func;

    code_ifthenelset if_throw;
    if_throw.cond() = not_exprt(match);
    if_throw.then_case() = throw_call;

    code_blockt block;
    block.copy_to_operands(if_throw);
    block.copy_to_operands(code_expressiont(cast_value));

    side_effect_exprt stmt_expr("statement_expression", ref_type);
    stmt_expr.copy_to_operands(block);
    new_expr = stmt_expr;
    return false;
  }

  exprt cast_or_null;
  if (arms.empty())
  {
    // No reachable D can satisfy the cast — emit a typed null directly.
    cast_or_null = typed_null;
  }
  else if (to_void)
  {
    // Per-D results differ; chain ITEs so each D selects its own offset.
    cast_or_null = typed_null;
    for (auto it = arms.rbegin(); it != arms.rend(); ++it)
      cast_or_null = if_exprt(
        equality_exprt(vptr_read, it->first), it->second, cast_or_null);
  }
  else
  {
    // T* form: every matching D yields the same `(T*) src`, so collapse
    // the per-D vptr equalities into one OR-chain with a single result.
    cast_or_null = if_exprt(vptr_match_any(), arms.front().second, typed_null);
  }

  // Null-source guard — without this, src_deref above would crash
  // symbolic execution before the SMT solver ever sees the ITE.
  exprt is_null = equality_exprt(src_pointer, gen_zero(src_pointer.type()));
  new_expr = if_exprt(is_null, typed_null, cast_or_null);
  return false;
}
