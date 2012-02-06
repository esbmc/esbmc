#include "renaming.h"

std::string renaming::level1t::name(
  const irep_idt &identifier,
  unsigned frame, unsigned execution_node_id) const
{
  return id2string(identifier)+"@"+i2string(frame)+"!"+i2string(_thread_id);//+"*"+i2string(execution_node_id);
}

unsigned renaming::level2t::current_number(
  const irep_idt &identifier) const
{
  current_namest::const_iterator it=current_names.find(identifier);
  if(it==current_names.end()) return 0;
  return it->second.count;
}

std::string renaming::level1t::operator()(
  const irep_idt &identifier, unsigned exec_node_id) const
{

  current_namest::const_iterator it=
    current_names.find(identifier);

  if(it==current_names.end())
  {
    // can not find
    return id2string(identifier); // means global value ?
  }

  return name(identifier, it->second, exec_node_id);
}

std::string renaming::level2t::operator()(
  const irep_idt &identifier, unsigned exec_node_id) const
{
  current_namest::const_iterator it=
    current_names.find(identifier);

  if(it==current_names.end())
    return name(identifier, 0);

  return name(identifier, it->second.count);
}

std::string renaming::level2t::stupid_operator(
  const irep_idt &identifier, unsigned exec_node_id) const
{
  current_namest::const_iterator it=
    current_names.find(identifier);

  if(it==current_names.end())
    return name(identifier, 0);

  return name(identifier, it->second.count);
}
