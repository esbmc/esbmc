#include <util/array2string.h>
#include <util/bitvector.h>

bool array2string(const symbolt &src, exprt &dest)
{
  if (src.get_type().id() != irept::id_array)
    return true;

  if (bv_width(src.get_type().subtype()) != 8) // TODO: handle wide strings
    return true;

  std::string value_str;
  size_t cnt = 0;
  forall_operands (it, src.get_value())
  {
    std::string op_str = it->cformat().as_string();
    if (cnt < src.get_value().operands().size() - 1)
      value_str.push_back(op_str[1]);
    cnt++;
  }

  exprt new_expr("string-constant", src.get_type());
  new_expr.value(value_str);
  dest = new_expr;
  return false;
}
