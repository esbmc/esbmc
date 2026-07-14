#include <util/exception_specification.h>
#include <util/type.h>

exception_specificationt exception_specificationt::from_type(const typet &type)
{
  exception_specificationt spec;

  const irept &kind_irep = type.find(kind_attribute());
  if (kind_irep.is_nil())
    return spec; // potentially_throwing

  const irep_idt &kind_id = kind_irep.id();
  if (kind_id == "non_throwing")
    spec.kind = kindt::non_throwing;
  else if (kind_id == "dynamic")
  {
    spec.kind = kindt::dynamic;
    const irept &types = type.find(types_attribute());
    for (const auto &sub : types.get_sub())
      spec.allowed_types.emplace_back(sub.id());
  }

  return spec;
}
