#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/cbmc_adapter.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program_irep.h>
#include <util/migrate.h>
#include <util/message.h>
#include <util/symbol.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>

bool is_cbmc_goto_magic(const unsigned char header[4])
{
  return header[0] == 0x7f && header[1] == 'G' && header[2] == 'B' &&
         header[3] == 'F';
}

unsigned cbmc_irep_readert::read_word()
{
  unsigned shift_distance = 0;
  // Accumulate in 64 bits so that bits shifted past bit 31 stay visible and an
  // over-wide varint is detectable rather than silently truncated.
  uint64_t res = 0;

  while (in.good())
  {
    if (shift_distance >= 64)
    {
      log_error("CBMC goto-binary: malformed varint (too many bytes)");
      abort();
    }

    unsigned byte = static_cast<unsigned char>(in.get());
    res |= static_cast<uint64_t>(byte & 0x7f) << shift_distance;
    shift_distance += 7;

    if (res > UINT32_MAX)
    {
      log_error("CBMC goto-binary: input number {} exceeds 32 bits", res);
      abort();
    }

    if ((byte & 0x80) == 0)
      break;
  }

  return static_cast<unsigned>(res);
}

std::string cbmc_irep_readert::read_string()
{
  std::string result;

  while (in.good())
  {
    int c = in.get();
    if (c == 0 || c == EOF)
      break;
    if (c == '\\') // escaped char
    {
      int e = in.get();
      if (e == EOF)
        break;
      result.push_back(static_cast<char>(e));
    }
    else
      result.push_back(static_cast<char>(c));
  }

  return result;
}

irep_idt cbmc_irep_readert::read_string_ref()
{
  unsigned id = read_word();

  auto it = string_cache.find(id);
  if (it != string_cache.end())
    return it->second;

  irep_idt s(read_string());
  string_cache.emplace(id, s);
  return s;
}

void cbmc_irep_readert::read_reference(irept &irep)
{
  unsigned id = read_word();

  auto it = irep_cache.find(id);
  if (it != irep_cache.end())
  {
    irep = it->second;
    return;
  }

  read_irep(irep);
  irep_cache.emplace(id, irep);
}

void cbmc_irep_readert::read_irep(irept &irep)
{
  irep.id(read_string_ref());

  while (in.peek() == 'S')
  {
    in.get();
    irep.get_sub().emplace_back();
    read_reference(irep.get_sub().back());
  }

  while (in.peek() == 'N')
  {
    in.get();
    irep_idt name = read_string_ref();
    read_reference(irep.add(name));
  }

  while (in.peek() == 'C')
  {
    in.get();
    irep_idt name = read_string_ref();
    read_reference(irep.add(name));
  }

  if (in.get() != 0)
  {
    log_error("CBMC goto-binary: irep not terminated");
    abort();
  }
}

bool parse_cbmc_goto(
  std::istream &in,
  const std::string &filename,
  cbmc_parse_resultt &result)
{
  // Header: 0x7f 'G' 'B' 'F'
  unsigned char hdr[4];
  for (unsigned char &b : hdr)
    b = static_cast<unsigned char>(in.get());

  if (!is_cbmc_goto_magic(hdr))
  {
    log_error("`{}' is not a CBMC goto-binary", filename);
    return true;
  }

  cbmc_irep_readert reader(in);

  unsigned version = reader.read_word();
  if (version != 6)
  {
    log_error(
      "unsupported CBMC goto-binary version {} (only 6 is supported)", version);
    return true;
  }

  // Symbol table
  unsigned number_of_symbols = reader.read_word();
  result.symbols.reserve(number_of_symbols);
  for (unsigned i = 0; i < number_of_symbols; i++)
  {
    cbmc_symbolt sym;
    reader.read_reference(sym.stype);
    reader.read_reference(sym.value);
    reader.read_reference(sym.location);
    sym.name = id2string(reader.read_string_ref());
    sym.module = id2string(reader.read_string_ref());
    sym.base_name = id2string(reader.read_string_ref());
    sym.mode = id2string(reader.read_string_ref());
    sym.pretty_name = id2string(reader.read_string_ref());

    // Ordering is kept for historical reasons and must be zero.
    unsigned ordering = reader.read_word();
    if (ordering != 0)
    {
      log_error("CBMC goto-binary: unexpected symbol ordering {}", ordering);
      return true;
    }

    sym.flags = reader.read_word();
    sym.is_type = (sym.flags & (1u << 15)) != 0;
    sym.is_weak = (sym.flags & (1u << 16)) != 0;
    sym.is_property = (sym.flags & (1u << 14)) != 0;
    sym.is_macro = (sym.flags & (1u << 13)) != 0;
    sym.is_exported = (sym.flags & (1u << 12)) != 0;
    sym.is_input = (sym.flags & (1u << 11)) != 0;
    sym.is_output = (sym.flags & (1u << 10)) != 0;
    sym.is_state_var = (sym.flags & (1u << 9)) != 0;
    sym.is_parameter = (sym.flags & (1u << 8)) != 0;
    sym.is_auxiliary = (sym.flags & (1u << 7)) != 0;
    sym.binding = (sym.flags & (1u << 6)) != 0;
    sym.is_lvalue = (sym.flags & (1u << 5)) != 0;
    sym.is_static_lifetime = (sym.flags & (1u << 4)) != 0;
    sym.is_thread_local = (sym.flags & (1u << 3)) != 0;
    sym.is_file_local = (sym.flags & (1u << 2)) != 0;
    sym.is_extern = (sym.flags & (1u << 1)) != 0;
    sym.is_volatile = (sym.flags & 1u) != 0;

    result.symbols.push_back(std::move(sym));
  }

  // Functions
  unsigned number_of_functions = reader.read_word();
  result.functions.reserve(number_of_functions);
  for (unsigned i = 0; i < number_of_functions; i++)
  {
    cbmc_functiont function;
    function.name = reader.read_string(); // raw string, not a string ref
    unsigned number_of_instructions = reader.read_word();
    function.instructions.reserve(number_of_instructions);

    for (unsigned j = 0; j < number_of_instructions; j++)
    {
      cbmc_instructiont instruction;
      reader.read_reference(instruction.code);
      reader.read_reference(instruction.source_location);
      instruction.instr_type = reader.read_word();
      reader.read_reference(instruction.guard);
      instruction.target_number = reader.read_word();

      unsigned target_count = reader.read_word();
      instruction.targets.reserve(target_count);
      for (unsigned t = 0; t < target_count; t++)
        instruction.targets.push_back(reader.read_word());

      unsigned label_count = reader.read_word();
      instruction.labels.reserve(label_count);
      for (unsigned l = 0; l < label_count; l++)
        instruction.labels.push_back(id2string(reader.read_string_ref()));

      function.instructions.push_back(std::move(instruction));
    }

    result.functions.push_back(std::move(function));
  }

  return false;
}

bool read_cbmc_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  goto_functionst &goto_functions)
{
  cbmc_parse_resultt parsed;
  if (parse_cbmc_goto(in, filename, parsed))
    return true;

  cbmc_adapted_resultt adapted = adapt_cbmc_to_esbmc(std::move(parsed));

  // Symbol table — mirrors read_bin_goto_object's symbol loop.
  for (const irept &t : adapted.symbols)
  {
    symbolt symbol;
    symbol.from_irep(t);

    if (!symbol.is_type && symbol.get_type().is_code())
    {
      // Ensure an (empty) function exists for every function symbol and fix
      // up the function type.
      auto it = goto_functions.function_map.find(symbol.id);
      if (it == goto_functions.function_map.end())
        goto_functions.function_map.emplace(symbol.id, goto_functiont());
      goto_functions.function_map.at(symbol.id).type =
        migrate_symbol_type(symbol);
    }

    context.add(symbol);
  }

  assert(migrate_namespace_lookup);

  // Function bodies — mirrors read_bin_goto_object's function loop.
  for (auto &named : adapted.functions)
  {
    const irep_idt fname = named.first;
    auto it = goto_functions.function_map.find(fname);
    if (it == goto_functions.function_map.end())
      goto_functions.function_map.emplace(fname, goto_functiont());
    goto_functiont &f = goto_functions.function_map.at(fname);
    convert(named.second, f.body);
    f.body_available = f.body.instructions.size() > 0;
  }

  return false;
}
