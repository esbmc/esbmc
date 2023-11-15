/*******************************************************************
 Module: Typecast fuzz tests

 Author: Rafael Sá Menezes

 Date: January 2021

 Fuzz Plan:
   - ToUnion (builtins)
 \*******************************************************************/

// ToUnion

/**
 * The idea is to initialize an union_typet with N types
 * identifying them by an integer. After that randomly select
 * two indexes, to confirm the types and to confirm the exception
 *
 */

#include <clang-c-frontend/typecast.h>
#include <util/type.h>
#include <util/expr_util.h>
#include "../testing-utils/util_irep.h"
#include <sstream>
#include <stddef.h>

#include <clang-c-frontend/clang_c_convert.h>

namespace
{
void gen_typecast_to_union(exprt &dest, const typet &type)
{
  contextt ctx;
  namespacet ns(ctx);
  clang_c_convertert::gen_typecast_to_union(ns, dest, type);
}
} // namespace

void test_to_union(const int *Data, size_t Size)
{
  union_typet t;

  // Get the index where the expected value will be
  int rand_index = Data[Size - 1] % 40;

  bool added_check = false;
  Builtin_Type bt_check = (Builtin_Type)(Data[0]);
  auto component_check = gen_component("special", bt_check);

  Builtin_Type bt_error = (Builtin_Type)(Data[1]);

  // Don't need to worry with upper bound (libFuzzer)
  for(size_t i = 2; i < Size - 1; i++)
  {
    if(Data[i] == Data[1])
      continue;
    if(!added_check && (Data[i] == Data[0] || i > (size_t)rand_index))
    {
      t.components().push_back(component_check);
      added_check = true;
      continue;
    }
    std::ostringstream name;
    name << "var_" << i;
    auto component = gen_component(name.str().c_str(), (Builtin_Type)(Data[i]));
    t.components().push_back(component);
  }

  // It could happen that Data[1] until Data[-1]
  // are the same, so it would never add the check
  if(!added_check)
    return;

  typet builtin;
  gen_builtin_type(builtin, bt_check);
  exprt e = gen_zero(builtin);

  gen_typecast_to_union(e, t);

  assert(to_union_expr(e).get_component_name() == component_check.name());
  assert(to_union_expr(e).op0().type() == component_check.type());

  try
  {
    typet error_builtin;
    gen_builtin_type(error_builtin, bt_error);
    exprt error = gen_zero(error_builtin);
    //Trigger exception
    gen_typecast_to_union(error, t);
    // This shouldn't be reached
    assert(0);
  }
  catch(const std::domain_error &)
  {
    // OK
    return;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const int *Data, size_t Size)
{
  if(Size < 5 || Size > 100 || (Data[0] == Data[1]))
    return 0;
  for(size_t i = 0; i < Size; i++)
  {
    if(Data[i] <= (int)Builtin_Type::Void || Data[i] >= (int)Builtin_Type::Last)
    {
      return 0;
    }
  }

  test_to_union(Data, Size);
  return 0;
}
