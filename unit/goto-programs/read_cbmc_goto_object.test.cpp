/*******************************************************************
 Module: CBMC goto-binary reader unit tests

 Covers the low-level reader ported from goto-transcoder's bytereader.rs
 + cbmc.rs: the 7-bit varint word encoding, backslash string escaping,
 header/version validation, and a full parse of a real CBMC v6 binary.

\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/migrate.h>
#include <util/namespace.h>

#include <fstream>
#include <sstream>
#include <string>

#ifndef CBMC_TEST_DATA_DIR
#  define CBMC_TEST_DATA_DIR "."
#endif

// Encode a value the way CBMC does: little-endian 7-bit varint.
static std::string encode_varint(unsigned v)
{
  std::string out;
  do
  {
    unsigned char byte = v & 0x7f;
    v >>= 7;
    if (v != 0)
      byte |= 0x80;
    out.push_back(static_cast<char>(byte));
  } while (v != 0);
  return out;
}

TEST_CASE("varint word decoding round-trips", "[cbmc-reader]")
{
  for (unsigned v :
       {0u, 1u, 0x7fu, 0x80u, 0x81u, 0x3fffu, 0x4000u, 12345u, 0xffffffffu})
  {
    std::istringstream in(encode_varint(v), std::ios::binary);
    cbmc_irep_readert reader(in);
    REQUIRE(reader.read_word() == v);
  }
}

TEST_CASE("varint matches known CBMC encodings", "[cbmc-reader]")
{
  // 39 -> single byte 0x27 (as seen in the real fixture header).
  REQUIRE(encode_varint(39) == std::string("\x27", 1));
  // 300 -> 0xAC 0x02
  REQUIRE(encode_varint(300) == std::string("\xac\x02", 2));
}

TEST_CASE("string reading unescapes backslash-escaped bytes", "[cbmc-reader]")
{
  // "a\0b\\c" encoded with CBMC escaping: a, \ 0, b, \ \, c, terminator 0.
  std::string raw;
  raw.push_back('a');
  raw.push_back('\\');
  raw.push_back('\0'); // escaped NUL
  raw.push_back('b');
  raw.push_back('\\');
  raw.push_back('\\'); // escaped backslash
  raw.push_back('c');
  raw.push_back('\0'); // terminator

  std::istringstream in(raw, std::ios::binary);
  cbmc_irep_readert reader(in);
  std::string s = reader.read_string();

  std::string expected;
  expected.push_back('a');
  expected.push_back('\0');
  expected.push_back('b');
  expected.push_back('\\');
  expected.push_back('c');
  REQUIRE(s == expected);
}

TEST_CASE("rejects a non-CBMC header", "[cbmc-reader]")
{
  // ESBMC's own format starts with 'G' 'B' 'F' (no 0x7f prefix).
  std::string bytes = "GBF";
  bytes.push_back('\0');
  std::istringstream in(bytes, std::ios::binary);
  cbmc_parse_resultt result;
  REQUIRE(parse_cbmc_goto(in, "fake.goto", result) == true);
}

TEST_CASE("parses a real CBMC v6 goto-binary", "[cbmc-reader]")
{
  const std::string path = std::string(CBMC_TEST_DATA_DIR) + "/cbmc_hello.goto";
  std::ifstream in(path, std::ios::in | std::ios::binary);
  REQUIRE(in.good());

  cbmc_parse_resultt result;
  bool failed = parse_cbmc_goto(in, path, result);

  REQUIRE_FALSE(failed);
  REQUIRE_FALSE(result.symbols.empty());
  REQUIRE_FALSE(result.functions.empty());

  // The fixture header advertises 39 symbols (varint 0x27).
  REQUIRE(result.symbols.size() == 39);

  // Every function name should be non-empty and every parsed instruction
  // should carry a code/guard irep (default-constructed ireps are "nil"-less
  // empty ids until read, so a populated id means the reference parsed).
  bool saw_instruction = false;
  for (const auto &f : result.functions)
  {
    REQUIRE_FALSE(f.name.empty());
    for (const auto &ins : f.instructions)
    {
      saw_instruction = true;
      // instr_type is a small enum value; sanity bound.
      REQUIRE(ins.instr_type < 64);
    }
  }
  REQUIRE(saw_instruction);
}

TEST_CASE(
  "loads a CBMC binary into a symbol table and goto functions",
  "[cbmc-reader]")
{
  const std::string path = std::string(CBMC_TEST_DATA_DIR) + "/cbmc_hello.goto";
  std::ifstream in(path, std::ios::in | std::ios::binary);
  REQUIRE(in.good());

  contextt context;
  goto_functionst goto_functions;
  namespacet ns(context);
  migrate_namespace_lookup = &ns;

  const bool failed = read_cbmc_goto_object(in, path, context, goto_functions);
  REQUIRE_FALSE(failed);

  // The symbol table got populated, including main.
  REQUIRE(context.find_symbol("main") != nullptr);

  // main has a converted body, and CBMC's entry point is present.
  auto it = goto_functions.function_map.find("main");
  REQUIRE(it != goto_functions.function_map.end());
  REQUIRE(it->second.body_available);
  REQUIRE(
    goto_functions.function_map.find("__CPROVER__start") !=
    goto_functions.function_map.end());

  migrate_namespace_lookup = nullptr;
}
