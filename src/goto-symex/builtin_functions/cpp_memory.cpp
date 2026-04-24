#include <cassert>
#include <complex>
#include <functional>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/printf_formatter.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_types.h>
#include <vector>
#include <algorithm>
#include <util/array2string.h>

void goto_symext::symex_cpp_new(
  const expr2tc &lhs,
  const sideeffect2t &code,
  const guardt &guard)
{
  expr2tc size = code.size;

  bool do_array = (code.kind == sideeffect2t::cpp_new_arr);

  unsigned int &dynamic_counter = get_dynamic_counter();
  dynamic_counter++;

  const std::string count_string(i2string(dynamic_counter));

  // value
  symbolt symbol;
  symbol.name = do_array ? "dynamic_" + count_string + "_array"
                         : "dynamic_" + count_string + "_value";
  symbol.id = "symex_dynamic::" + id2string(symbol.name);
  symbol.lvalue = true;
  symbol.mode = "C++";

  const pointer_type2t &ptr_ref = to_pointer_type(code.type);
  type2tc renamedtype2 =
    migrate_type(ns.follow(migrate_type_back(ptr_ref.subtype)));

  type2tc newtype = do_array
                      ? type2tc(array_type2tc(renamedtype2, code.size, false))
                      : renamedtype2;

  symbol.type = migrate_type_back(newtype);

  symbol.type.dynamic(true);

  new_context.add(symbol);

  // make symbol expression
  expr2tc rhs_ptr_obj;
  if (do_array)
  {
    expr2tc sym = symbol2tc(newtype, symbol.id);
    expr2tc idx = index2tc(renamedtype2, sym, gen_ulong(0));
    rhs_ptr_obj = idx;
  }
  else
    rhs_ptr_obj = symbol2tc(newtype, symbol.id);

  expr2tc rhs = address_of2tc(renamedtype2, rhs_ptr_obj);

  cur_state->rename(rhs);
  expr2tc rhs_copy(rhs);
  expr2tc ptr_rhs(rhs);

  symex_assign(code_assign2tc(lhs, rhs), true);

  expr2tc ptr_obj = pointer_object2tc(pointer_type2(), ptr_rhs);
  track_new_pointer(ptr_obj, newtype, guard, size);

  guardt g(cur_state->guard);
  g.append(guard);
  dynamic_memory.emplace_back(rhs_copy, g, false, symbol.name.as_string());
}

void goto_symext::symex_cpp_delete(const expr2tc &expr)
{
  const auto &code = static_cast<const code_expression_data &>(*expr);

  expr2tc tmp = code.operand;

  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_empty_type(), tmp);
  dereference(deref, dereferencet::INTERNAL);

  // we need to check the memory deallocation operator:
  // new and delete, new[] and delete[]
  if (internal_deref_items.size())
  {
    bool is_arr = is_array_type(internal_deref_items.front().object->type);
    bool is_del_arr = is_code_cpp_del_array2t(expr);

    if (is_arr != is_del_arr)
    {
      const std::string &msg =
        "Mismatched memory deallocation operators: " + get_expr_id(expr);
      claim(gen_false_expr(), msg);
    }
  }
  // implement delete as a call to free
  symex_free(expr);
}
