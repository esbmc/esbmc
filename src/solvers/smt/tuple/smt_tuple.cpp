#include <solvers/smt/tuple/smt_tuple.h>
#include <irep2/irep2_type.h>
#include <util/message/message.h>
#include <string>

void check_tuple_field(
  unsigned int idx,
  std::size_t size,
  const type2tc &tuple_type)
{
  if (idx < size)
    return;

  log_error(
    "Tuple field {} is out of range: the tuple for {} holds {} field(s), so "
    "it disagrees with the expression type this index came from",
    idx,
    struct_union_name(tuple_type),
    size);
  throw std::string("tuple field out of range");
}
