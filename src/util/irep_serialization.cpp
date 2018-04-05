/*******************************************************************\
 
Module: binary irep conversions with hashing
 
Author: CM Wintersteiger
 
Date: May 2007
 
\*******************************************************************/

#include <sstream>
#include <util/irep_serialization.h>
#include <util/string_hash.h>

void irep_serializationt::write_irep(std::ostream &out, const irept &irep)
{
  write_string_ref(out, irep.id_string());

  forall_irep(it, irep.get_sub())
  {
    out.put('S');
    reference_convert(*it, out);
  }

  forall_named_irep(it, irep.get_named_sub())
  {
    out.put('N');
    write_string_ref(out, name2string(it->first));
    reference_convert(it->second, out);
  }

  forall_named_irep(it, irep.get_comments())
  {
    out.put('C');
    write_string_ref(out, name2string(it->first));
    reference_convert(it->second, out);
  }

  out.put(0); // terminator
}

void irep_serializationt::reference_convert(std::istream &in, irept &irep)
{
  unsigned id = read_long(in);

  if(
    ireps_container.ireps_on_read.find(id) !=
    ireps_container.ireps_on_read.end())
  {
    irep = ireps_container.ireps_on_read[id];
  }
  else
  {
    read_irep(in, irep);
    ireps_container.ireps_on_read[id] = irep;
  }
}

void irep_serializationt::read_irep(std::istream &in, irept &irep)
{
  irep.id(read_string_ref(in));

  while(in.peek() == 'S')
  {
    in.get();
    irep.get_sub().emplace_back();
    reference_convert(in, irep.get_sub().back());
  }

  while(in.peek() == 'N')
  {
    in.get();
    irept &r = irep.add(read_string_ref(in));
    reference_convert(in, r);
  }

  while(in.peek() == 'C')
  {
    in.get();
    irept &r = irep.add(read_string_ref(in));
    reference_convert(in, r);
  }

  if(in.get() != 0)
  {
    std::cerr << "irep not terminated. " << std::endl;
    throw 0;
  }
}

void irep_serializationt::reference_convert(
  const irept &irep,
  std::ostream &out)
{
  // Do we have this irep already? Horrible complexity here.
  unsigned int i;
  for(i = 0; i < ireps_container.ireps_on_write.size(); i++)
  {
    if(full_eq(ireps_container.ireps_on_write[i], irep))
    {
      // Match, at idx i
      write_long(out, i);
      return;
    }
  }

  i = ireps_container.ireps_on_write.size();
  ireps_container.ireps_on_write.push_back(irep);
  write_long(out, i);
  write_irep(out, irep);
}

void write_long(std::ostream &out, unsigned u)
{
  out.put((u & 0xFF000000) >> 24);
  out.put((u & 0x00FF0000) >> 16);
  out.put((u & 0x0000FF00) >> 8);
  out.put(u & 0x000000FF);
}

unsigned irep_serializationt::read_long(std::istream &in)
{
  unsigned res = 0;

  for(unsigned i = 0; i < 4 && in.good(); i++)
    res = (res << 8) | in.get();

  return res;
}

void write_string(std::ostream &out, const std::string &s)
{
  for(char i : s)
  {
    if(i == 0 || i == '\\')
      out.put('\\'); // escape specials
    out << i;
  }

  out.put(0);
}

dstring irep_serializationt::read_string(std::istream &in)
{
  char c;
  unsigned i = 0;

  while((c = in.get()) != 0)
  {
    if(i >= read_buffer.size())
      read_buffer.resize(read_buffer.size() * 2, 0);
    if(c == '\\') // escaped chars
      read_buffer[i] = in.get();
    else
      read_buffer[i] = c;
    i++;
  }

  if(i >= read_buffer.size())
    read_buffer.resize(read_buffer.size() * 2, 0);
  read_buffer[i] = 0;

  return dstring(&(read_buffer[0]));
}

void irep_serializationt::write_string_ref(std::ostream &out, const dstring &s)
{
  unsigned id = s.get_no();
  if(id >= ireps_container.string_map.size())
    ireps_container.string_map.resize(id + 1, false);

  if(ireps_container.string_map[id])
    write_long(out, id);
  else
  {
    ireps_container.string_map[id] = true;
    write_long(out, id);
    write_string(out, s.as_string());
  }
}

irep_idt irep_serializationt::read_string_ref(std::istream &in)
{
  unsigned id = read_long(in);

  if(id >= ireps_container.string_rev_map.size())
    ireps_container.string_rev_map.resize(
      1 + id * 2, std::pair<bool, dstring>(false, dstring()));
  if(ireps_container.string_rev_map[id].first)
  {
    return ireps_container.string_rev_map[id].second;
  }

  dstring s = read_string(in);
  ireps_container.string_rev_map[id] = std::pair<bool, dstring>(true, s);
  return ireps_container.string_rev_map[id].second;
}
