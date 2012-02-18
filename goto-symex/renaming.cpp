#include "renaming.h"

std::string renaming::level1t::name(
  const irep_idt &identifier,
  unsigned frame, unsigned execution_node_id) const
{
  return id2string(identifier)+"@"+i2string(frame)+"!"+i2string(level1_data._thread_id);//+"*"+i2string(execution_node_id);
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

  level1_datat::current_namest::const_iterator it=
    level1_data.current_names.find(identifier);

  if(it==level1_data.current_names.end())
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

static std::string state_to_ignore[8] =
{"\\guard", "trds_count", "trds_in_run", "deadlock_wait", "deadlock_mutex",
"count_lock", "count_wait", "unlocked"};

crypto_hash
renaming::level2t::generate_l2_state_hash() const
{
  unsigned int total;

  uint8_t *data = (uint8_t*)alloca(current_hashes.size() * CRYPTO_HASH_SIZE * sizeof(uint8_t));

  total = 0;
  for (current_state_hashest::const_iterator it = current_hashes.begin();
        it != current_hashes.end(); it++) {
    int j;

    for (j = 0 ; j < 8; j++)
      if (it->first.as_string().find(state_to_ignore[j]) != std::string::npos)
        continue;

    memcpy(&data[total * CRYPTO_HASH_SIZE], it->second.hash, CRYPTO_HASH_SIZE);
    total++;
  }

  return crypto_hash(data, total * CRYPTO_HASH_SIZE);
}

void renaming::level1t::rename(exprt &expr,unsigned node_id)
{
  // rename all the symbols with their last known value

  rename(expr.type(),node_id);

  if(expr.id()==exprt::symbol)
  {
    const irep_idt &identifier=expr.identifier();

    // first see if it's already an l1 name

    if(renaming_data.original_identifiers.find(identifier)!=
       renaming_data.original_identifiers.end())
      return;

    const level1_datat::current_namest::const_iterator it=
      level1_data.current_names.find(identifier);

    if(it!=level1_data.current_names.end())
      expr.identifier(name(identifier, it->second,node_id));
  }
  else if(expr.id()==exprt::addrof ||
          expr.id()=="implicit_address_of" ||
          expr.id()=="reference_to")
  {
    assert(expr.operands().size()==1);
    rename(expr.op0(),node_id);
  }
  else
  {
    // do this recursively
    Forall_operands(it, expr)
      rename(*it,node_id);
  }
}

void renaming::level2t::rename(exprt &expr, unsigned node_id)
{
  // rename all the symbols with their last known value

  rename(expr.type(),node_id);

  if(expr.id()==exprt::symbol)
  {
    const irep_idt &identifier=expr.identifier();

    // first see if it's already an l2 name

    if(renaming_data.original_identifiers.find(identifier)!=
       renaming_data.original_identifiers.end())
      return;

    const current_namest::const_iterator it=
      current_names.find(identifier);

    if(it!=current_names.end())
    {
      if(it->second.constant.is_not_nil())
        expr=it->second.constant;
      else
        expr.identifier(name(identifier, it->second.count));
    }
    else
    {
      std::string new_identifier=name(identifier, 0);
      renaming_data.original_identifiers[new_identifier]=identifier;
      expr.identifier(new_identifier);
    }
  }
  else if(expr.id()==exprt::addrof ||
          expr.id()=="implicit_address_of" ||
          expr.id()=="reference_to")
  {
    // do nothing
  }
  else
  {
    // do this recursively
    Forall_operands(it, expr)
      rename(*it,node_id);
  }
}

void renaming::renaming_levelt::rename(typet &type, unsigned node_id)
{
  // rename all the symbols with their last known value

  if(type.id()==typet::t_array)
  {
    rename(type.subtype(),node_id);
    exprt tmp = static_cast<const exprt &>(type.size_irep());
    rename(tmp, node_id);
    type.size(tmp);
  }
  else if(type.id()==typet::t_struct ||
          type.id()==typet::t_union ||
          type.id()==typet::t_class)
  {
    // TODO
  }
  else if(type.id()==typet::t_pointer)
  {
    rename(type.subtype(),node_id);
  }
}

void renaming::renaming_levelt::get_original_name(exprt &expr) const
{
  Forall_operands(it, expr)
    get_original_name(*it);

  if(expr.id()==exprt::symbol)
  {
    original_identifierst::const_iterator it=
      renaming_data.original_identifiers.find(expr.identifier());
    if(it==renaming_data.original_identifiers.end()) return;

    assert(it->second!="");
    expr.identifier(it->second);
  }
}

const irep_idt &renaming::renaming_levelt::get_original_name(
  const irep_idt &identifier) const
{
  original_identifierst::const_iterator it=
    renaming_data.original_identifiers.find(identifier);
  if(it==renaming_data.original_identifiers.end()) return identifier;
  return it->second;
}

void renaming::level1t::print(std::ostream &out,unsigned node_id) const
{
  for(level1_datat::current_namest::const_iterator
      it=level1_data.current_names.begin();
      it!=level1_data.current_names.end();
      it++)
    out << it->first << " --> "
        << name(it->first, it->second,node_id) << std::endl;
}

void renaming::level2t::print(std::ostream &out, unsigned node_id) const
{
  for(current_namest::const_iterator
      it=current_names.begin();
      it!=current_names.end();
      it++)
    out << it->first << " --> "
        << name(it->first, it->second.count) << std::endl;

}
