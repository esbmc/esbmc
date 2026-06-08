#include <util/exception_specification.h>
#include <util/type.h>

bool exception_specificationt::allows(const irep_idt &exception_type_id) const
{
  switch (kind)
  {
  case kindt::potentially_throwing:
    return true;
  case kindt::non_throwing:
    return false;
  case kindt::dynamic:
    for (const auto &allowed : allowed_types)
      if (allowed == exception_type_id)
        return true;
    return false;
  }
  return true;
}

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

void exception_specificationt::to_type(typet &type) const
{
  switch (kind)
  {
  case kindt::potentially_throwing:
    return; // no metadata: this is the default
  case kindt::non_throwing:
    type.set(kind_attribute(), "non_throwing");
    return;
  case kindt::dynamic:
  {
    type.set(kind_attribute(), "dynamic");
    irept &types = type.add(types_attribute());
    types.get_sub().clear();
    for (const auto &allowed : allowed_types)
    {
      irept entry;
      entry.id(allowed);
      types.get_sub().push_back(entry);
    }
    return;
  }
  }
}
