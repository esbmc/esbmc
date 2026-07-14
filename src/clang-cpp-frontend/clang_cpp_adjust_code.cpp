#include <functional>

#include <clang-cpp-frontend/clang_cpp_adjust.h>
#include <clang-c-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/exception_specification.h>
#include <util/message.h>
#include <util/std_expr.h>

void clang_cpp_adjust::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

void clang_cpp_adjust::adjust_code(codet &code)
{
  const irep_idt &statement = code.statement();

  if (statement == "cpp-catch")
  {
    adjust_catch(code);
  }
  else
    clang_c_adjust::adjust_code(code);
}

void clang_cpp_adjust::adjust_ifthenelse(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // if(void *p=...) ...
  if (code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_ifthenelse(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_ifthenelse(code);
}

void clang_cpp_adjust::adjust_while(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // while(void *p=...) ...
  if (code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_while(code);

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_while(code);
}

void clang_cpp_adjust::adjust_switch_case_ops(
  exprt &expr,
  const typet &switch_type)
{
  if (!expr.is_code())
    return;

  codet &c = to_code(expr);
  if (c.get_statement() == "switch_case")
  {
    code_switch_caset &sc = to_code_switch_case(c);
    if (!sc.is_default() && sc.case_op().type() != switch_type)
      gen_typecast(ns, sc.case_op(), switch_type);

    adjust_switch_case_ops(sc.code(), switch_type);
    return;
  }

  // Don't recurse into nested switch statements
  if (c.get_statement() == "switch")
    return;

  Forall_operands (it, c)
    adjust_switch_case_ops(*it, switch_type);
}

void clang_cpp_adjust::adjust_switch(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // switch(int i=...) ...
  if (code.op0().is_code())
  {
    codet decl_block = to_code(code.op0());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // replace declaration by its symbol
    code_declt decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());

    code.op0() = decl.op0();
    clang_c_adjust::adjust_switch(code);

    adjust_switch_case_ops(code.op1(), code.op0().type());

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block.op0(), code);
    code.swap(code_block);
  }
  else
  {
    clang_c_adjust::adjust_switch(code);
    adjust_switch_case_ops(code.op1(), code.op0().type());
  }
}

void clang_cpp_adjust::adjust_for(codet &code)
{
  // In addition to the C syntax, C++ also allows a declaration
  // as condition. E.g.,
  // for( ; int i=...; ) ...
  if (code.op1().is_code())
  {
    codet decl_block = to_code(code.op1());

    assert(decl_block.get_statement() == "decl-block");
    assert(decl_block.operands().size() == 1);

    adjust_code(decl_block);

    // Create new cond assignment
    code_declt &decl = to_code_decl(to_code(decl_block.op0()));
    assert(decl.op0().is_symbol());
    assert(decl.operands().size() == 2);

    side_effect_exprt new_cond("assign", decl.op0().type());
    new_cond.copy_to_operands(decl.op0(), decl.op1());
    adjust_expr(new_cond);

    code.op1() = new_cond;
    clang_c_adjust::adjust_for(code);

    // Remove assignment
    decl.operands().pop_back();

    // Create new block
    code_blockt code_block;
    code_block.move_to_operands(decl_block, code);
    code.swap(code_block);
  }
  else
    clang_c_adjust::adjust_for(code);
}

// Recursively locate the constructor-call side-effect inside a declaration
// initializer.  Mirrors find_constructor_call in clang_cpp_main.cpp: for a
// `temporary_object` the call lives under the named "initializer" sub-irep
// (not an operand), so unwrap that before recursing into operands.
static const exprt *find_constructor_call(const exprt &e)
{
  if (
    e.id() == "sideeffect" && e.statement() == "function_call" &&
    e.get_bool("constructor"))
    return &e;

  if (e.id() == "sideeffect" && e.statement() == "temporary_object")
  {
    const irept &init = e.initializer();
    if (init.is_not_nil())
      if (
        const exprt *c =
          find_constructor_call(static_cast<const exprt &>(init)))
        return c;
  }

  forall_operands (it, e)
    if (const exprt *c = find_constructor_call(*it))
      return c;

  return nullptr;
}

void clang_cpp_adjust::adjust_decl_block(codet &code)
{
  codet new_block("decl-block");

  Forall_operands (it, code)
  {
    if (it->is_code() && it->statement() == "skip")
      continue;

    adjust_expr(*it);
    code_declt &code_decl = to_code_decl(to_code(*it));

    // A local (automatic-storage) array of class type with a non-trivial
    // constructor must have *every* element constructed.  The frontend leaves
    // a single constructor call on the declaration and relies on
    // clang_cpp_maint::adjust_init to expand it per element -- but that only
    // runs for static-storage objects, so a local array would construct only
    // element 0.  Fan the call out here into one constructor call per element
    // (recursing into nested arrays), each with `this` pointing at that
    // element.  See regression esbmc-cpp11/constructors/Constructor9-1.
    // Only whole-array default/value construction qualifies: the initializer
    // is a *single* constructor call (wrapped in a temporary_object) whose
    // type is the whole array.  Aggregate initialisation such as
    // `B a[2] = {B(1), B(2)}` lowers to a constant_array of per-element
    // initialisers instead, and must NOT be fanned out (that would construct
    // every element with element 0's arguments).
    // Static-storage locals (function-local statics, e.g. `static B a[2];`)
    // are constructed by static_lifetime_init, not from the function body, so
    // leave their declaration untouched -- expanding here would construct them
    // a second time on every call.
    const bool is_static_local = code_decl.op0().is_symbol() && [&] {
      const symbolt *s =
        ns.lookup(to_symbol_expr(code_decl.op0()).get_identifier());
      return s && s->static_lifetime;
    }();

    const exprt *ctor = nullptr;
    if (
      !is_static_local && code_decl.operands().size() == 2 &&
      ns.follow(code_decl.op0().type()).is_array())
    {
      const exprt &init = code_decl.op1();
      const bool single_ctor_init =
        init.id() == "sideeffect" &&
        (init.statement() == "temporary_object" ||
         (init.statement() == "function_call" && init.get_bool("constructor")));
      if (single_ctor_init)
        ctor = find_constructor_call(init);
    }
    if (ctor)
    {
      const exprt array = code_decl.op0();
      const side_effect_expr_function_callt call =
        to_side_effect_expr_function_call(*ctor);

      // Emit the bare declaration (no initializer) first.
      code_declt bare_decl(code_decl.op0());
      bare_decl.location() = code_decl.location();
      new_block.copy_to_operands(bare_decl);

      std::function<void(const exprt &)> construct_elements =
        [&](const exprt &arr) {
          const array_typet &arr_type = to_array_type(ns.follow(arr.type()));
          BigInt count;
          if (to_integer(arr_type.size(), count))
          {
            log_error("cannot determine array size for local ctor init");
            abort();
          }

          const typet &elem_type = arr_type.subtype();
          for (BigInt idx = 0; idx < count; ++idx)
          {
            exprt element =
              index_exprt(arr, from_integer(idx, index_type()), elem_type);
            if (ns.follow(elem_type).is_array())
            {
              construct_elements(element);
              continue;
            }
            side_effect_expr_function_callt elem_call = call;
            elem_call.arguments()[0] = address_of_exprt(element);
            exprt as_code = elem_call;
            convert_expression_to_code(as_code);
            new_block.copy_to_operands(as_code);
          }
        };
      construct_elements(array);
      continue;
    }

    new_block.copy_to_operands(code_decl);
  }

  code.swap(new_block);
}

void clang_cpp_adjust::adjust_catch(codet &code)
{
  codet::operandst &operands = code.operands();
  // adjust try block
  adjust_expr(operands[0]);

  // First operand is always the try block, skip it
  for (auto it = ++operands.begin(); it != operands.end(); it++)
  {
    // The following operands are the catchs
    adjust_expr(*it);
    code_blockt &block = to_code_block(to_code(*it));

    std::vector<irep_idt> ids;
    convert_exception_id(block.type(), "", ids);

    block.type() = code_typet();
    block.set("exception_id", ids.front());
  }
}

void clang_cpp_adjust::finalize_exception_specification(typet &type)
{
  if (
    type.get(exception_specificationt::kind_attribute()) != "dynamic" ||
    type.find("exception_spec_decl").is_nil())
    return;

  // Resolve each declared exception type to its exception id. As in the old
  // throw_decl handling, we keep only the leading id per declared type (the
  // literal type itself); base classes are expanded at the throw site.
  irept &decl = type.add("exception_spec_decl");
  irept resolved;
  for (const auto &op : decl.get_sub())
  {
    std::vector<irep_idt> ids;
    convert_exception_id(static_cast<const typet &>(op), "", ids);
    if (!ids.empty())
    {
      irept entry;
      entry.id(ids.front());
      resolved.get_sub().push_back(entry);
    }
  }

  type.set(exception_specificationt::types_attribute(), resolved);
  type.remove("exception_spec_decl");
}
