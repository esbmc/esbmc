/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_scope.h>
#include <cpp/cpp_typecheck.h>

void cpp_scopet::lookup(
  const irep_idt &base_name,
  id_sett &id_set)
{
  cpp_id_mapt::iterator
    lower_it=sub.lower_bound(base_name);

  if(lower_it!=sub.end())
  {
    cpp_id_mapt::iterator
      upper_it=sub.upper_bound(base_name);

    for(cpp_id_mapt::iterator n_it=lower_it;
        n_it!=upper_it; n_it++)
      id_set.insert(&n_it->second);
  }

  if(this->base_name == base_name)
    id_set.insert(this);

  for(unsigned i =0; i< parents_size(); i++)
  {
    cpp_idt& parent= get_parent(i);
    if(parent.base_name == base_name)
      id_set.insert(&parent);
  }

  // using directives
  for(auto it : using_set)
  {
    cpp_idt& using_id = *it;
    if(using_id.base_name == base_name)
      id_set.insert(it);

    if(using_id.is_scope)
    {
      ((cpp_scopet&)using_id).lookup(base_name,id_set);
    }
  }
}

void cpp_scopet::recursive_lookup(
  const irep_idt &base_name,
  id_sett &id_set)
{
  lookup(base_name, id_set);

  // found nothing? Ask parent
  if(id_set.empty())
  {
    for(unsigned i= 0; i < parents_size(); i++)
      get_parent(i).recursive_lookup(base_name, id_set); // recursive call
  }
}



void cpp_scopet::lookup(
  const irep_idt &base_name,
  cpp_idt::id_classt id_class,
  id_sett &id_set)
{
  cpp_id_mapt::iterator
    lower_it=sub.lower_bound(base_name);

  if(lower_it!=sub.end())
  {
    cpp_id_mapt::iterator
      upper_it=sub.upper_bound(base_name);

    for(cpp_id_mapt::iterator n_it=lower_it;
        n_it!=upper_it; n_it++)
    {
      if(n_it->second.id_class == id_class)
        id_set.insert(&n_it->second);
    }
  }

  if(this->base_name == base_name
     && this->id_class == id_class)
    id_set.insert(this);

  for(unsigned i =0; i< parents_size(); i++)
  {
    cpp_idt& parent= get_parent(i);
    if(parent.base_name == base_name
       && parent.id_class == id_class)
        id_set.insert(&parent);
  }

  // using directives
  for(auto it : using_set)
  {
    cpp_idt& using_id = *it;
    if(using_id.base_name == base_name && using_id.id_class == id_class)
      id_set.insert(it);

    if(using_id.is_scope)
    {
      ((cpp_scopet&)using_id).lookup(base_name, id_class, id_set);
    }
  }
}

void cpp_scopet::recursive_lookup(
  const irep_idt &base_name,
  cpp_idt::id_classt id_class,
  id_sett &id_set)
{
  lookup(base_name, id_class, id_set);

  // found nothing? Ask parent
  if(id_set.empty() && parents_size())
      get_parent().recursive_lookup(base_name, id_class, id_set); // recursive call
}

void cpp_scopet::lookup_id(
  const irep_idt &identifier,
  cpp_idt::id_classt id_class,
  id_sett &id_set)
{
  for(auto & n_it : sub)
  {
    if(n_it.second.identifier == identifier
       && n_it.second.id_class == id_class)
          id_set.insert(&n_it.second);
  }

  if(this->identifier == identifier
     && this->id_class == id_class)
    id_set.insert(this);

  for(unsigned i =0; i< parents_size(); i++)
  {
    cpp_idt& parent= get_parent(i);
    if(parent.identifier == identifier
       && parent.id_class == id_class)
        id_set.insert(&parent);
  }
}



cpp_scopet &cpp_scopet::new_scope(const irep_idt &new_scope_name)
{
  cpp_idt &id=insert(new_scope_name);
  id.identifier=prefix+id2string(new_scope_name);
  id.prefix=prefix+id2string(new_scope_name)+"::";
  id.this_expr=this_expr;
  id.class_identifier=class_identifier;
  id.is_scope=true;
  return (cpp_scopet &)id;
}


bool cpp_scopet::contains(const irep_idt& base_name)
{
  id_sett id_set;
  lookup(base_name,id_set);
  return !id_set.empty();
}
