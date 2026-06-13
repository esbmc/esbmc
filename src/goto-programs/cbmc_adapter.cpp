// Faithful C++ port of goto-transcoder's adapter.rs: rewrites CBMC irep
// conventions into ESBMC's so the result feeds symbolt::from_irep and the
// goto_program_irep convert() directly. Function/struct names and control flow
// mirror the Rust to keep the two implementations easy to diff.

#include <goto-programs/cbmc_adapter.h>
#include <util/message.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace
{
// Construct an id-only irep (mirrors Rust Irept::from(<string>)).
irept mk(const irep_idt &id)
{
  return irept(id);
}

// Mirrors Rust's `named_subt.contains_key`, honouring ESBMC's '#'-comment
// routing (all keys used here are non-'#', i.e. true named-subs).
bool has_sub(const irept &i, const irep_idt &k)
{
  // Mirror irept::is_comment: '#'-prefixed names live in the comments map.
  const std::string &ks = k.as_string();
  const irept::named_subt &m =
    (!ks.empty() && ks[0] == '#') ? i.get_comments() : i.get_named_sub();
  return m.find(k) != m.end();
}

// CBMC sometimes stores constant values as hex; ESBMC wants a binary string.
// Mirrors Rust `format!("{:032b}", u64::from_str_radix(value, 16))`.
std::string hex_to_bin32(const std::string &hex)
{
  unsigned long long n = std::stoull(hex, nullptr, 16);
  std::string bits;
  if (n == 0)
    bits = "0";
  else
    while (n != 0)
    {
      bits.push_back(static_cast<char>('0' + (n & 1)));
      n >>= 1;
    }
  std::reverse(bits.begin(), bits.end());
  if (bits.size() < 32)
    bits = std::string(32 - bits.size(), '0') + bits;
  return bits;
}

bool irep_contains(const irept &i, const irep_idt &id)
{
  if (i.id() == id)
    return true;
  for (const auto &s : i.get_sub())
    if (irep_contains(s, id))
      return true;
  for (const auto &p : i.get_named_sub())
    if (irep_contains(p.second, id))
      return true;
  for (const auto &p : i.get_comments())
    if (irep_contains(p.second, id))
      return true;
  return false;
}

void fix_expression(irept &irep)
{
  if (irep.id() == "side_effect")
    irep.id("sideeffect");
  else if (irep.id() == "string_constant")
    irep.id("string-constant");
  else if (irep.id() == "ieee_float_equal")
    irep.id("=");

  if (irep.id() == "constant")
  {
    const std::string type_id = irep.find("type").id_string();
    if (type_id != "pointer" && type_id != "bool" && has_sub(irep, "value"))
    {
      const std::string val = irep.find("value").id_string();
      // Value may be a hex representation or binary; we want the binary one.
      if (val.size() != 32)
        irep.add("value") = mk(hex_to_bin32(val));
    }
  }

  static const std::unordered_set<std::string> expressions = {
    "if",
    "member",
    "typecast",
    "notequal",
    "and",
    "or",
    "mod",
    "not",
    "*",
    "/",
    "+",
    "-",
    "=",
    "<",
    "<=",
    ">",
    ">=",
    "unary-",
    "overflow_result-+",
    "overflow_result--",
    "overflow_result-*",
    "overflow_result-shr",
    "lshr",
    "ashr",
    "shl",
    "address_of",
    "index",
    "byte_extract_little_endian",
    "pointer_object",
    "array_of",
    "sideeffect",
    "dereference",
    "object_size",
    "bitand",
    "bitor",
    "bitxor",
    "bitnot",
    "struct",
    "return",
    "r_ok"};

  const std::string cur = irep.id_string();

  // In CBMC both an expression and a type can be named "array"; "array" is an
  // umbrella expression that can also mean index.
  const bool array_has_operand = cur == "array" && has_sub(irep, "type") &&
                                 irep.find("type").id() == "array" &&
                                 !irep.get_sub().empty();

  const bool is_function_call = cur == "arguments" && !irep.get_sub().empty();

  if (expressions.count(cur) != 0 || array_has_operand || is_function_call)
  {
    irept operands;
    operands.get_sub() = irep.get_sub();
    irep.get_sub().clear();
    irep.add("operands") = operands;
  }

  for (auto &s : irep.get_sub())
    fix_expression(s);

  for (auto &p : irep.get_named_sub())
  {
    if (p.first == "components")
      for (auto &c : p.second.get_sub())
        c.id("component");
    fix_expression(p.second);
  }
}

void expand_anon_struct(const irept &self)
{
  if (has_sub(self, "components"))
    return;
  // ESBMC has no parser for CBMC's anonymous naming convention.
  const std::string ident = self.find("identifier").id_string();
  if (ident.size() < 11 || ident.compare(0, 10, "tag-#anon#") != 0)
    return;
  log_error("CBMC adapter: unsupported anonymous struct {}", ident);
  abort();
}

void fix_type(irept &self, const std::unordered_map<std::string, irept> &cache)
{
  if (self.id() == "c_bool")
  {
    self.id("signedbv");
    return;
  }

  if (self.id() == "code" && has_sub(self, "parameters"))
  {
    irept arguments(irep_idt("arguments"));
    arguments.get_sub() = self.find("parameters").get_sub();
    self.add("arguments") = arguments;
  }

  if (has_sub(self, "components"))
  {
    irept &components = self.add("components");
    for (auto &v : components.get_sub())
      v.id("component"); // fix_struct
  }

  if (self.id() == "pointer" && !has_sub(self, "subtype"))
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
    irept operands;
    operands.get_sub() = self.get_sub();
    self.add("subtype") = operands;
    self.get_sub().clear();
  }

  if (
    self.id() == "array" && !has_sub(self, "subtype") &&
    !self.get_sub().empty())
  {
    irept magic = self.get_sub()[0];
    self.add("subtype") = magic;
    self.get_sub().clear();
    // CBMC can't decide whether array sizes are binary or hex.
    for (auto &p : self.get_named_sub())
      if (p.first == "size" && has_sub(p.second, "value"))
        fix_expression(p.second);
  }

  if (self.id() != "struct_tag")
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache);
    for (auto &p : self.get_comments())
      fix_type(p.second, cache);
    return;
  }

  if (!has_sub(self, "identifier"))
    return;

  const std::string ident = self.find("identifier").id_string();
  auto it = cache.find(ident);
  if (it == cache.end())
  {
    expand_anon_struct(self);
    return;
  }

  self = it->second;

  // The resolved struct may itself contain struct tags; redo the cache walk.
  if (irep_contains(self, "struct_tag"))
  {
    for (auto &v : self.get_sub())
      fix_type(v, cache);
    for (auto &p : self.get_named_sub())
      fix_type(p.second, cache);
    for (auto &p : self.get_comments())
      fix_type(p.second, cache);
  }
}

irept symbol_to_esbmc_irep(const cbmc_symbolt &sym)
{
  irept result;
  result.add("type") = sym.stype;
  result.add("symvalue") = sym.value;
  result.add("location") = sym.location;
  result.add("module") = mk(sym.module);
  result.add("mode") = mk(sym.mode);

  // fix_name is the identity in adapter.rs.
  if (sym.is_type)
    result.add("is_type") = mk("1");
  if (sym.is_macro)
    result.add("is_macro") = mk("1");
  if (sym.is_parameter)
    result.add("is_parameter") = mk("1");
  if (sym.is_lvalue)
    result.add("lvalue") = mk("1");
  if (sym.is_static_lifetime)
    result.add("static_lifetime") = mk("1");
  if (sym.is_file_local)
    result.add("file_local") = mk("1");
  if (sym.is_extern)
    result.add("is_extern") = mk("1");

  // Flags ESBMC has no equivalent for are dropped (roadmap §4.5). Warn when one
  // is actually set, since volatile/thread_local in particular affect soundness.
  if (sym.is_volatile)
    log_warning("CBMC adapter: dropping 'volatile' on symbol {}", sym.name);
  if (sym.is_thread_local)
    log_warning("CBMC adapter: dropping 'thread_local' on symbol {}", sym.name);
  if (sym.is_weak)
    log_warning("CBMC adapter: dropping 'weak' on symbol {}", sym.name);
  if (sym.is_property)
    log_warning("CBMC adapter: dropping 'property' on symbol {}", sym.name);

  result.add("base_name") = mk(sym.base_name);
  result.add("name") = mk(sym.name);

  fix_expression(result);
  return result;
}

irept instruction_to_esbmc_irep(
  const cbmc_instructiont &ins,
  const std::map<unsigned, unsigned> &target_revmap,
  const std::string &function_name)
{
  irept result;

  if (ins.instr_type == 19)
  {
    log_error("CBMC adapter: unexpected instruction type 19");
    abort();
  }

  // ESBMC expects code arguments inside "operands".
  irept code = ins.code;
  irept operands;
  operands.get_sub() = code.get_sub();
  code.get_sub().clear();
  code.add("operands") = operands;

  if (code.id() != "nil" && code.find("statement").id() == "assign")
    if (code.find("operands").get_sub().size() != 2)
    {
      log_error("CBMC adapter: assign must have exactly two operands");
      abort();
    }

  result.add("code") = code;
  result.add("location") = ins.source_location;
  result.add("typeid") = mk(std::to_string(ins.instr_type));
  result.add("guard") = ins.guard;

  if (!ins.targets.empty())
  {
    irept t_ireps;
    for (unsigned raw : ins.targets)
    {
      auto it = target_revmap.find(raw);
      if (it == target_revmap.end())
      {
        log_error("CBMC adapter: unresolved target number {}", raw);
        abort();
      }
      t_ireps.get_sub().push_back(mk(std::to_string(it->second)));
    }
    result.add("targets") = t_ireps;
  }

  if (!ins.labels.empty())
  {
    irept l_ireps;
    for (const auto &label : ins.labels)
      l_ireps.get_sub().push_back(mk(label));
    result.add("labels") = l_ireps;
  }

  result.add("function") = mk(function_name);

  fix_expression(result);
  return result;
}

irept function_to_esbmc_irep(const cbmc_functiont &func)
{
  // CBMC numbers targets from 1 with a per-instruction target number; ESBMC
  // uses the instruction's position. Build the reverse map over ALL
  // instructions first (matching adapter.rs), then emit.
  std::map<unsigned, unsigned> target_revmap;
  for (unsigned i = 0; i < func.instructions.size(); i++)
    target_revmap[func.instructions[i].target_number] = i;

  irept result(irep_idt("goto-program"));
  for (const auto &ins : func.instructions)
  {
    const bool is_output =
      ins.code.id() != "nil" && ins.code.find("statement").id() == "output";
    if (!is_output)
      result.get_sub().push_back(
        instruction_to_esbmc_irep(ins, target_revmap, func.name));
  }
  return result;
}

} // namespace

cbmc_adapted_resultt adapt_cbmc_to_esbmc(cbmc_parse_resultt parsed)
{
  cbmc_adapted_resultt out;
  out.symbols.reserve(parsed.symbols.size());
  out.functions.reserve(parsed.functions.size());

  // First, map all struct ref-types into concrete types via a tag cache.
  std::unordered_map<std::string, irept> type_cache;

  for (auto &sym : parsed.symbols)
  {
    if (sym.is_type && sym.stype.id() == "struct")
    {
      const std::string tagname = "tag-" + sym.base_name;
      fix_type(sym.stype, type_cache);
      type_cache[tagname] = sym.stype;
    }
    out.symbols.push_back(symbol_to_esbmc_irep(sym));
  }

  // A symbol might be defined later; re-check every symbol with the full cache.
  for (auto &symbol : out.symbols)
  {
    fix_type(symbol, type_cache);
    if (irep_contains(symbol, "struct_tag"))
    {
      log_error("CBMC adapter: struct_tag should have been resolved");
      abort();
    }
    if (symbol.find("type").id() == "c_bool")
    {
      log_error("CBMC adapter: c_bool type should have been rewritten");
      abort();
    }
  }

  for (auto &func : parsed.functions)
  {
    irept function_irep = function_to_esbmc_irep(func);
    fix_type(function_irep, type_cache);
    out.functions.emplace_back(func.name, std::move(function_irep));
  }

  return out;
}
