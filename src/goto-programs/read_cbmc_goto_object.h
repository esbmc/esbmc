#pragma once

#include <istream>
#include <map>
#include <string>
#include <vector>
#include <util/irep.h>

class contextt;
class goto_functionst;

/// Low-level reader for the CBMC goto-binary format (version 6).
///
/// Mirrors \ref irep_serializationt but uses CBMC's 7-bit varint word encoding
/// instead of ESBMC's big-endian 32-bit words. The S/N/C reference grammar and
/// the string/irep reference caching are identical to ESBMC's reader.
class cbmc_irep_readert
{
public:
  explicit cbmc_irep_readert(std::istream &in) : in(in)
  {
  }

  /// CBMC encodes words as little-endian 7-bit varints (continuation bit 0x80).
  unsigned read_word();

  /// Reads a NUL-terminated string with backslash escaping (same as ESBMC).
  std::string read_string();

  /// Reads a word-tagged, cached string reference.
  irep_idt read_string_ref();

  /// Reads a word-tagged, cached irep reference (with S/N/C children).
  void read_reference(irept &dest);

private:
  void read_irep(irept &dest);

  std::istream &in;
  std::map<unsigned, irept> irep_cache;
  std::map<unsigned, irep_idt> string_cache;
};

/// Intermediate, pre-adaptation parse of a CBMC symbol (mirrors cbmc.rs
/// CBMCSymbol). Conventions are still CBMC's; the adapter rewrites them.
struct cbmc_symbolt
{
  irept stype; // CBMC "type"; renamed to avoid the C++ keyword clash Rust hit
  irept value;
  irept location;
  std::string name;
  std::string module;
  std::string base_name;
  std::string mode;
  std::string pretty_name;
  unsigned flags = 0;
  bool is_type = false;
  bool is_weak = false;
  bool is_property = false;
  bool is_macro = false;
  bool is_exported = false;
  bool is_input = false;
  bool is_output = false;
  bool is_state_var = false;
  bool is_parameter = false;
  bool is_auxiliary = false;
  bool binding = false;
  bool is_lvalue = false;
  bool is_static_lifetime = false;
  bool is_thread_local = false;
  bool is_file_local = false;
  bool is_extern = false;
  bool is_volatile = false;
};

/// Intermediate parse of a CBMC instruction (mirrors cbmc.rs CBMCInstruction).
/// Targets are kept as the raw CBMC target numbers; the adapter remaps them to
/// ESBMC's instruction-index scheme.
struct cbmc_instructiont
{
  irept code;
  irept source_location;
  irept guard;
  unsigned instr_type = 0;
  unsigned target_number = 0;
  std::vector<unsigned> targets;
  std::vector<std::string> labels;
};

struct cbmc_functiont
{
  std::string name;
  std::vector<cbmc_instructiont> instructions;
};

struct cbmc_parse_resultt
{
  std::vector<cbmc_symbolt> symbols;
  std::vector<cbmc_functiont> functions;
};

/// Parses a CBMC goto-binary stream (header 0x7f 'G' 'B' 'F', version 6) into
/// the intermediate, still-CBMC-convention structures above. Returns true on
/// error. Mirrors cbmc.rs process_cbmc_file.
bool parse_cbmc_goto(
  std::istream &in,
  const std::string &filename,
  cbmc_parse_resultt &result);

/// Reads a CBMC goto-binary, adapts it to ESBMC irep conventions, and populates
/// the symbol table and goto functions — the CBMC-format counterpart of
/// read_bin_goto_object. Returns true on error.
bool read_cbmc_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  goto_functionst &goto_functions);
