#include <new>
#include <stdexcept>
#include <sstream>
#include <util/irep/irep_serialization.h>
#include <util/message/message.h>

void irep_serializationt::write_irep(std::ostream &out, const irept &irep)
{
  write_string_ref(out, irep.id_string());

  forall_irep (it, irep.get_sub())
  {
    out.put('S');
    reference_convert(*it, out);
  }

  forall_named_irep (it, irep.get_named_sub())
  {
    out.put('N');
    write_string_ref(out, name2string(it->first));
    reference_convert(it->second, out);
  }

  forall_named_irep (it, irep.get_comments())
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
  auto &ireps_on_read = ireps_container.ireps_on_read;

  if (id < ireps_on_read.size())
  {
    irep = ireps_on_read[id];
    return;
  }

  /* A first occurrence always takes the next index; anything beyond that is a
   * forward reference the format cannot express, i.e. a corrupt stream. */
  if (id != ireps_on_read.size())
  {
    // Thrown, not abort()ed: this is a statement about the *input*, and
    // read_goto_binary's caller turns a throw into a graceful error exit. A
    // truncated file reaches here routinely.
    log_error("goto binary: irep {} referenced before it is defined", id);
    throw std::string("goto binary: irep referenced before it is defined");
  }

  ireps_on_read.emplace_back(); // claim the slot before the nested ids are read
  read_irep(in, irep);
  ireps_on_read[id] = irep;
}

void irep_serializationt::read_irep(std::istream &in, irept &irep)
{
  irep.id(read_string_ref(in));

  while (in.peek() == 'S')
  {
    in.get();
    irep.get_sub().emplace_back();
    reference_convert(in, irep.get_sub().back());
  }

  while (in.peek() == 'N')
  {
    in.get();
    irept &r = irep.add(read_string_ref(in));
    reference_convert(in, r);
  }

  while (in.peek() == 'C')
  {
    in.get();
    irept &r = irep.add(read_string_ref(in));
    reference_convert(in, r);
  }

  if (in.get() != 0)
    throw std::string("goto binary: irep not terminated");
}

void irep_serializationt::reference_convert(
  const irept &irep,
  std::ostream &out)
{
  // Do we have this irep already?
  unsigned i = ireps_container.ireps_on_write.size();
  auto [it, ins] = ireps_container.ireps_on_write.try_emplace(irep, i);
  write_long(out, it->second);
  if (ins)
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

  for (unsigned i = 0; i < 4; i++)
  {
    const int c = in.get();
    if (c == EOF)
    {
      // Leave the stream failed so the caller stops rather than building ireps
      // out of a partial word; returning the partial value silently produced
      // counts and ids that no longer describe the file.
      in.setstate(std::ios::failbit);
      return 0;
    }
    res = (res << 8) | static_cast<unsigned>(c);
  }

  return res;
}

void write_string(std::ostream &out, const std::string &s)
{
  for (char i : s)
  {
    if (i == 0 || i == '\\')
      out.put('\\'); // escape specials
    out << i;
  }

  out.put(0);
}

irep_idt irep_serializationt::read_string(std::istream &in)
{
  unsigned i = 0;

  // int, not char: get() signals end-of-input with EOF (-1), which narrows to
  // a perfectly ordinary char and never compares equal to the terminator. On a
  // truncated stream the loop therefore never ended, doubling read_buffer until
  // the process died -- a hang, not a diagnostic, on any short goto-binary.
  for (int c; (c = in.get()) != 0 && c != EOF;)
  {
    if (i >= read_buffer.size())
      read_buffer.resize(read_buffer.size() * 2, 0);
    if (c == '\\') // escaped chars
    {
      const int escaped = in.get();
      if (escaped == EOF)
      {
        // Same reasoning as read_long: a stream that ends mid-escape has no
        // character to store, and narrowing EOF would hand the caller a 0xff
        // the file never contained.
        in.setstate(std::ios::failbit);
        break;
      }
      read_buffer[i] = static_cast<char>(escaped);
    }
    else
      read_buffer[i] = static_cast<char>(c);
    i++;
  }

  if (i >= read_buffer.size())
    read_buffer.resize(read_buffer.size() * 2, 0);
  read_buffer[i] = 0;

  return irep_idt(&(read_buffer[0]));
}

void irep_serializationt::write_string_ref(std::ostream &out, const irep_idt &s)
{
  unsigned id = s.get_no();
  if (id >= ireps_container.string_map.size())
    ireps_container.string_map.resize(id + 1, false);

  if (ireps_container.string_map[id])
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

  if (id >= ireps_container.string_rev_map.size())
  {
    // `1 + id * 2` was computed in 32 bits: a corrupted id of 0x80000000
    // wrapped to resize(1), and the indexing below then ran off the end of the
    // map -- a SIGSEGV on any goto-binary with a flipped byte here.
    //
    // The id is deliberately not checked against the input length. It is the
    // writer's *string-pool* number (write_string_ref stores
    // irep_idt::get_no()), not a dense per-stream counter, so a small binary
    // produced by a process that has interned many strings carries ids far
    // past its own size; bounding by the file rejects valid input. A corrupt
    // id instead surfaces as a table this allocator cannot serve.
    try
    {
      ireps_container.string_rev_map.resize(
        1 + static_cast<std::size_t>(id) * 2,
        std::pair<bool, irep_idt>(false, irep_idt()));
    }
    catch (const std::bad_alloc &)
    {
      log_error("goto binary: string id {} needs an unservable table", id);
      throw std::string("goto binary: implausible string id");
    }
    catch (const std::length_error &)
    {
      log_error("goto binary: string id {} needs an unservable table", id);
      throw std::string("goto binary: implausible string id");
    }
  }
  if (ireps_container.string_rev_map[id].first)
  {
    return ireps_container.string_rev_map[id].second;
  }

  irep_idt s = read_string(in);
  ireps_container.string_rev_map[id] = std::pair<bool, irep_idt>(true, s);
  return ireps_container.string_rev_map[id].second;
}
