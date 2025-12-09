#include <irep2/irep2_expr.h>

#include <util/arith_tools.h>
#include <util/config.h>
#include <util/migrate.h>
#include <util/std_types.h>
#include <util/string2array.h>

void string2array(const exprt &src, exprt &dest)
{
  expr2tc src2;
  migrate_expr(src, src2);
  expr2tc arr = to_constant_string2t(src2).to_array();
  dest = migrate_expr_back(arr);

  for (exprt &op : dest.operands())
  {
    BigInt ch = binary2integer(op.value().as_string(), false);
    if (ch >= 32 && ch <= 126)
    {
      char ch_str[2];
      ch_str[0] = ch.to_uint64();
      ch_str[1] = 0;

      op.cformat("'" + std::string(ch_str) + "'");
    }
  }
}
