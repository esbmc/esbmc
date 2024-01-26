#include <util/array2string.h>
#include <util/bitvector.h>

bool array2string(const symbolt &src, exprt &dest)
{
  if (src.type.id() != irept::id_array)
    return true;

  if (bv_width(src.type.subtype()) != 8) // TODO: handle wide strings
    return true;

  std::string value_str;
  size_t cnt = 0;
  forall_operands (it, src.value)
  {
    std::string op_str = it->cformat().as_string();
    if (cnt < src.value.operands().size() - 1)
      value_str.push_back(op_str[1]);
    cnt++;
  }

  exprt new_expr("string-constant", src.type);
  new_expr.value(value_str);
  dest = new_expr;
  return false;
}
