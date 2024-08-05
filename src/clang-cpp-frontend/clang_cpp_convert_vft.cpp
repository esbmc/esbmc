/*
 * This file contains functions to generate virtual function table (VFT):
 *  - travere virtual methods
 *  - generate VFT type symbol
 *  - generate VFT variable symbols
 *  - generate thunk functions for overriding methods
 */
#include "clang_cpp_vft_gen.h"

#include <util/compiler_defs.h>
// Remove warnings from Clang headers
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Basic/Version.inc>
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
#include <util/expr_util.h>
#include <util/message.h>

bool clang_cpp_convertert::get_struct_class_virtual_methods(
  const clang::CXXRecordDecl &cxxrd,
  struct_typet &type)
{
  for (const auto &md : cxxrd.methods())
  {
    if (!md->isVirtual())
      continue;
    type.set("#has_vptr_component", true);

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

      for (const auto &[_, overridden_method] : cxxmethods_overriden)
      {
        std::string base_id, base_name;
        get_decl_name(*overridden_method.getParent(), base_name, base_id);
        add_thunk_method(base_id, comp, type);
      }
    }
  }

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

void clang_cpp_convertert::add_thunk_method(
  const std::string &base_class_id,
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
  add_thunk_method_body(thunk_func_symb, component);

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
  const struct_typet::componentt &component)
{
  code_typet &code_type = to_code_type(thunk_func_symb.type);
  code_typet::argumentst &args = code_type.arguments();

  /*
   * late cast of `this` pointer to (Derived*)this
   */
  typecast_exprt late_cast_this(
    to_code_type(component.type()).arguments()[0].type());
  late_cast_this.op0() =
    symbol_expr(*namespacet(context).lookup(args[0].cmt_identifier()));

  if (
    code_type.return_type().id() != "empty" &&
    code_type.return_type().id() != "destructor")
    add_thunk_method_body_return(thunk_func_symb, component, late_cast_this);
  else
    add_thunk_method_body_no_return(thunk_func_symb, component, late_cast_this);
}

void clang_cpp_convertert::add_thunk_method_body_return(
  symbolt &thunk_func_symb,
  const struct_typet::componentt &component,
  const typecast_exprt &late_cast_this)
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
  const typecast_exprt &late_cast_this)
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
