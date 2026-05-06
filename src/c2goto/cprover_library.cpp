#include <c2goto/cprover_library.h>

#include "util/symbolic_types.h"

#include <ac_config.h>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fstream>
#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/message.h>
#include <util/c_link.h>
#include <util/config.h>
#include <util/language.h>

extern "C"
{
  extern const uint8_t clib32_buf[];
  extern const uint8_t clib64_buf[];
  extern const unsigned int clib32_buf_size;
  extern const unsigned int clib64_buf_size;

  extern const uint8_t clib32_fp_buf[];
  extern const uint8_t clib64_fp_buf[];
  extern const unsigned int clib32_fp_buf_size;
  extern const unsigned int clib64_fp_buf_size;

  extern const uint8_t clib32_cherih_buf[];
  extern const uint8_t clib64_cherih_buf[];
  extern const unsigned int clib32_cherih_buf_size;
  extern const unsigned int clib64_cherih_buf_size;

  extern const uint8_t clib32_fp_cherih_buf[];
  extern const uint8_t clib64_fp_cherih_buf[];
  extern const unsigned int clib32_fp_cherih_buf_size;
  extern const unsigned int clib64_fp_cherih_buf_size;

  extern const uint8_t clib32_cherip_buf[];
  extern const uint8_t clib64_cherip_buf[];
  extern const unsigned int clib32_cherip_buf_size;
  extern const unsigned int clib64_cherip_buf_size;

  extern const uint8_t clib32_fp_cherip_buf[];
  extern const uint8_t clib64_fp_cherip_buf[];
  extern const unsigned int clib32_fp_cherip_buf_size;
  extern const unsigned int clib64_fp_cherip_buf_size;

#ifdef ENABLE_SOLIDITY_FRONTEND
  extern const uint8_t sol64_buf[];
  extern const unsigned int sol64_buf_size;
#endif
}

namespace
{
/* [cheri][floatbv ? 1 : 0][wordsz == 64 ? 1 : 0] */
static const struct buffer
{
  const uint8_t *start;
  size_t size;
} clibs[3][2][2] = {
#ifdef ESBMC_BUNDLE_LIBC
  {
    {
#  ifdef ESBMC_BUNDLE_LIBC_32BIT
      {&clib32_buf[0], clib32_buf_size},
#  else
      {NULL, 0},
#  endif
      {&clib64_buf[0], clib64_buf_size},
    },
    {
#  ifdef ESBMC_BUNDLE_LIBC_32BIT
      {&clib32_fp_buf[0], clib32_fp_buf_size},
#  else
      {NULL, 0},
#  endif
      {&clib64_fp_buf[0], clib64_fp_buf_size},
    },
  },
  {
#  ifdef ESBMC_CHERI_HYBRID_SYSROOT
    {
      {NULL, 0}, // {&clib32_cherih_buf[0], clib32_cherih_buf_size},
      {&clib64_cherih_buf[0], clib64_cherih_buf_size},
    },
    {
      {NULL, 0}, // {&clib32_fp_cherih_buf[0], clib32_fp_cherih_buf_size},
      {&clib64_fp_cherih_buf[0], clib64_fp_cherih_buf_size},
    },
#  endif
  },
  {
#  ifdef ESBMC_CHERI_PURECAP_SYSROOT
    {
      {NULL, 0}, // {&clib32_cherip_buf[0], clib32_cherip_buf_size},
      {&clib64_cherip_buf[0], clib64_cherip_buf_size},
    },
    {
      {NULL, 0}, // {&clib32_fp_cherip_buf[0], clib32_fp_cherip_buf_size},
      {&clib64_fp_cherip_buf[0], clib64_fp_cherip_buf_size},
    },
#  endif
  },
#endif
};

// The goto reader will only pick up symbols for these functions and their dependencies
// This is a Python-specific whitelist invoked when you use set_functions_to_read
const static std::vector<std::string> python_c_models = {
  "__ESBMC_list_create",
  "list_in_bounds",
  "__ESBMC_list_at",
  "__ESBMC_list_clear",
  "__ESBMC_list_push",
  "__ESBMC_list_extend",
  "__ESBMC_list_insert",
  "__ESBMC_list_push_object",
  "__ESBMC_list_size",
  "list_hash_string",
  "__ESBMC_list_eq",
  "__ESBMC_list_set_eq",
  "strncmp",
  "strcmp",
  "strlen",
  "ceil",
  "__ceil_array",
  "fegetround",
  "fesetround",
  "rint",
  "fesetround",
  "floor",
  "fabs",
  "sin",
  "cos",
  "exp",
  "expm1",
  "expm1_taylor",
  "exp2",
  "fmod",
  "sqrt",
  "fmin",
  "fmax",
  "trunc",
  "frexp",
  "round",
  "copysign",
  "tan",
  "asin",
  "sinh",
  "cosh",
  "tanh",
  "log10",
  "arctan",
  "atan",
  "_atan",
  "atan2",
  "acos",
  "arccos",
  "dot",
  "add",
  "subtract",
  "multiply",
  "divide",
  "transpose",
  "det",
  "matmul",
  "pow",
  "log",
  "pow_by_squaring",
  "log2",
  "log1p",
  "ldexp",
  "log1p_taylor",
  "asinh",
  "acosh",
  "atanh",
  "hypot",
  "strstr",
  "strchr",
  "__ESBMC_list_contains",
  "__python_str_isdigit",
  "__python_char_isdigit",
  "__python_str_isalpha",
  "__python_char_isalpha",
  "__python_str_isspace",
  "isspace",
  "__python_str_lstrip",
  "__python_str_rstrip",
  "__python_str_strip",
  "__python_str_lstrip_chars",
  "__python_str_rstrip_chars",
  "__python_str_strip_chars",
  "__python_char_islower",
  "__python_str_islower",
  "__python_char_lower",
  "__python_str_lower",
  "__python_char_upper",
  "__python_str_upper",
  "__python_str_find",
  "__python_str_find_range",
  "__python_str_rfind",
  "__python_str_rfind_range",
  "__python_str_replace",
  "__python_str_split",
  "__ESBMC_create_inf_obj",
  "__python_int",
  "__python_chr",
  "__python_str_concat",
  "__python_str_repeat",
  "__python_str_slice",
  "__ESBMC_list_find_index",
  "__ESBMC_list_remove_at",
  "__ESBMC_list_set_at",
  "__ESBMC_list_pop",
  "__ESBMC_list_try_find_index",
  "__ESBMC_dict_eq",
  "__ESBMC_sin",
  "__ESBMC_cos",
  "__ESBMC_sqrt",
  "__ESBMC_exp",
  "__ESBMC_log",
  "__ESBMC_list_copy",
  "__ESBMC_tan",
  "__ESBMC_asin",
  "__ESBMC_sinh",
  "__ESBMC_cosh",
  "__ESBMC_tanh",
  "__ESBMC_log10",
  "__ESBMC_inf",
  "__ESBMC_nan",
  "__ESBMC_expm1",
  "__ESBMC_log1p",
  "__ESBMC_exp2",
  "__ESBMC_asinh",
  "__ESBMC_acosh",
  "__ESBMC_atanh",
  "__ESBMC_hypot",
  "__ESBMC_acos",
  "__ESBMC_atan",
  "__ESBMC_atan2",
  "__ESBMC_log2",
  "__ESBMC_pow",
  "__ESBMC_fabs",
  "__ESBMC_trunc",
  "__ESBMC_fmod",
  "__ESBMC_copysign",
  "__ESBMC_list_remove",
  "__ESBMC_list_sort",
  "__ESBMC_list_reverse",
  "__ESBMC_list_push_dict_ptr",
  "__ESBMC_list_lt"};

// Solidity operational model functions
const static std::vector<std::string> solidity_c_models = {
  // blockchain (solidity_blockchain.c)
  "blockhash",
  "blobhash",
  "gasConsume",
  "gasleft",
  // builtins (solidity_builtins.c)
  "sol_pow_uint",
  "addmod",
  "mulmod",
  "llc_nondet_bytes",
  "selfdestruct",
  // abi (solidity_abi.c)
  "abi_encode",
  "abi_encodePacked",
  "abi_encodeWithSelector",
  "abi_encodeWithSignature",
  "abi_encodeCall",
  "abi_decode",
  // crypto (solidity_crypto.c)
  "keccak256",
  "sha256",
  "ripemd160",
  "ecrecover",
  "string_concat",
  // bytes (solidity_bytes.c)
  "bytes_dynamic_init_check",
  "bytes_dynamic_bounds_check",
  "hex_char_to_nibble",
  "bytes_static_from_hex",
  "bytes_static_from_string",
  "bytes_static_truncate",
  "bytes_static_and",
  "bytes_static_or",
  "bytes_static_xor",
  "bytes_static_to_uint",
  "bytes_static_from_uint",
  "bytes_static_shl",
  "bytes_static_shr",
  "bytes_static_to_mapping_key",
  "bytes_static_init_zero",
  "bytes_static_set",
  "bytes_static_get",
  "bytes_static_equal",
  "bytes_static_to_string",
  "bytes_static_extend",
  "bytes_static_resize",
  "bytes_static_extend_from_dynamic",
  "bytes_static_resize_from_dynamic",
  "bytes_static_truncate_from_dynamic",
  "bytes_dynamic_init_zero",
  "bytes_dynamic_init",
  "bytes_dynamic_ensure_capacity",
  "bytes_dynamic_from_static",
  "bytes_dynamic_from_string",
  "bytes_dynamic_from_hex",
  "bytes_dynamic_concat",
  "bytes_dynamic_copy",
  "bytes_dynamic_set",
  "bytes_dynamic_get",
  "bytes_dynamic_equal",
  "bytes_dynamic_to_mapping_key",
  "bytes_dynamic_push",
  "bytes_dynamic_pop",
  "bytes_dynamic_to_uint",
  "bytes_dynamic_to_string",
  "bytes_pool_init",
  // mapping (solidity_mapping.c)
  "map_get_raw",
  "map_set_raw",
  "map_uint_set",
  "map_uint_get",
  "map_int_set",
  "map_int_get",
  "map_string_set",
  "map_string_get",
  "map_bool_set",
  "map_bool_get",
  "map_generic_set",
  "map_generic_get",
  "map_get_raw_fast",
  "map_set_raw_fast",
  "map_uint_set_fast",
  "map_uint_get_fast",
  "map_int_set_fast",
  "map_int_get_fast",
  "map_string_set_fast",
  "map_string_get_fast",
  "map_bool_set_fast",
  "map_bool_get_fast",
  "map_generic_set_fast",
  "map_generic_get_fast",
  // array (solidity_array.c)
  "_ESBMC_array_null_check",
  "_ESBMC_element_null_check",
  "_ESBMC_zero_size_check",
  "_ESBMC_pop_empty_check",
  "_ESBMC_store_array",
  "_ESBMC_array_length",
  "_ESBMC_arrcpy",
  "_ESBMC_array_push",
  "_ESBMC_array_pop",
  // units (solidity_units.c)
  "_ESBMC_wei",
  "_ESBMC_gwei",
  "_ESBMC_szabo",
  "_ESBMC_finney",
  "_ESBMC_ether",
  "_ESBMC_seconds",
  "_ESBMC_minutes",
  "_ESBMC_hours",
  "_ESBMC_days",
  "_ESBMC_weeks",
  "_ESBMC_years",
  // string (solidity_string.c)
  "get_char",
  "sol_rev",
  "i256toa",
  "u256toa",
  "decToHexa",
  "ASCIItoHEX",
  "hexdec",
  "str2uint",
  "_ESBMC_str_key_fold64",
  "_str_assign",
  "nondet_string",
  // address (solidity_address.c)
  "_ESBMC_get_addr_array_idx",
  "_ESBMC_cmp_cname",
  "_ESBMC_get_obj",
  "update_addr_obj",
  "_ESBMC_get_unique_address",
  "_ESBMC_get_nondet_cont_name",
  // misc (solidity_misc.c)
  "_max",
  "_min",
  "_creationCode",
  "_runtimeCode",
  "_interfaceId",
  "_ESBMC_check_reentrancy",
  "initialize"};
} // namespace

static void generate_symbol_deps(
  irep_idt name,
  irept irep,
  std::multimap<irep_idt, irep_idt> &deps)
{
  std::pair<irep_idt, irep_idt> type;

  if (irep.id() == "symbol")
  {
    type = std::pair<irep_idt, irep_idt>(name, irep.identifier());
    deps.insert(type);

    /* Cannot return here just yet
     * Symbol identifier may point to variable identifier
     * Further traversal needed to find type identifier if exists
     */
  }

  forall_irep (irep_it, irep.get_sub())
  {
    if (irep_it->id() == "argument")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->cmt_identifier());
      deps.insert(type);
    }
    else
    {
      /* Even if symbol & identifier found, further traversal might be needed for type identifier
       * Continue traversing to find symbol dependencies
       * The subcall will add the symbol identifier before traversing named and unnamed ireps so does not need to be done explicitly here
       */
      generate_symbol_deps(name, *irep_it, deps);
    }
  }

  /* The case where symbol identifier is reached but there are more nested type symbols
   * has only been seen so far when these higher-level symbols are unnamed ireps.
   *        (in particular inside an "operands" named_irep the layer above that)
   * Therefore named_irep iterator should be able to terminate on named_irep symbols
   * If there are future symbol resolution issues, consider changing this to also keep traversing
   *        For debugging, you can look at the nested structure via irept::pretty()
   */
  forall_named_irep (irep_it, irep.get_named_sub())
  {
    if (irep_it->second.id() == "symbol")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->second.identifier());
      deps.insert(type);
    }
    else if (irep_it->second.id() == "argument")
    {
      type =
        std::pair<irep_idt, irep_idt>(name, irep_it->second.cmt_identifier());
      deps.insert(type);
    }
    else
    {
      generate_symbol_deps(name, irep_it->second, deps);
    }
  }
}

static void ingest_symbol(
  irep_idt name,
  std::multimap<irep_idt, irep_idt> &deps,
  std::list<irep_idt> &to_include)
{
  std::pair<
    std::multimap<irep_idt, irep_idt>::const_iterator,
    std::multimap<irep_idt, irep_idt>::const_iterator>
    range;
  std::multimap<irep_idt, irep_idt>::const_iterator it;

  range = deps.equal_range(name);
  if (range.first == range.second)
    return;

  for (it = range.first; it != range.second; ++it)
    to_include.push_back(it->second);

  deps.erase(name);
}

void add_cprover_library(contextt &context, const languaget *language)
{
  if (config.ansi_c.lib == configt::ansi_ct::libt::LIB_NONE)
    return;

  contextt new_ctx, store_ctx;
  goto_functionst goto_functions;
  std::multimap<irep_idt, irep_idt> symbol_deps;
  std::list<irep_idt> to_include;
  const buffer *clib;

  switch (config.ansi_c.word_size)
  {
  case 16:
    log_warning(
      "this version of ESBMC does not have a C library for 16 bit machines");
    return;
  case 32:
#ifndef ESBMC_BUNDLE_LIBC_32BIT
    log_warning(
      "this version of ESBMC does not have a C library for 32 bit machines");
    return;
#endif
  case 64:
    break;
  default:
    log_error("No C library for bitwidth {}", config.ansi_c.word_size);
    abort();
  }

  clib = &clibs[config.ansi_c.cheri][!config.ansi_c.use_fixed_for_float]
               [config.ansi_c.word_size == 64];

  if (clib->size == 0)
  {
    if (language)
    {
      // C library sources must be parsed with C frontend, not the current language
      std::unique_ptr<languaget> c_lang(new_language(language_idt::C));
      return add_bundled_library_sources(context, *c_lang);
    }
    log_error("Zero-lengthed internal C library");
    abort();
  }

  goto_binary_reader goto_reader;

  if (language && language->id() == "python")
    goto_reader.set_functions_to_read(python_c_models);

  // Solidity uses a separate, smaller goto binary (sol64) for fast loading.
  // No whitelist needed: sol64 contains ONLY Solidity symbols.
  const uint8_t *lib_start;
  unsigned int lib_size;
  bool is_solidity = false;
#ifdef ENABLE_SOLIDITY_FRONTEND
  if (language && language->id() == "solidity_ast")
  {
    lib_start = sol64_buf;
    lib_size = sol64_buf_size;
    is_solidity = true;
  }
  else
#endif
  {
    if (language && language->id() == "python")
      goto_reader.set_functions_to_read(python_c_models);
    lib_start = clib->start;
    lib_size = clib->size;
  }

  /* Python: actively has a function filter
   *    - not everything makes it into new_ctx
   *    - ignored symbols go into ignored_ctx
   * Other languages: no function filter
   *    - everything makes it into new_ctx
   *    - ignored_ctx empty
   */
  contextt ignored_ctx;
  if (goto_reader.read_goto_binary_array(
        lib_start, lib_size, new_ctx, ignored_ctx, goto_functions))
    abort();

  // Traverse symbols and get dependencies from both their nested types and values
  new_ctx.foreach_operand([&symbol_deps](const symbolt &s) {
    generate_symbol_deps(s.id, s.value, symbol_deps);
    generate_symbol_deps(s.id, s.type, symbol_deps);
  });

  // Add two hacks; we might use either pthread_mutex_lock or the checked
  // variant; so if one version is used, pull in the other too.
  std::pair<irep_idt, irep_idt> lockcheck(
    dstring("pthread_mutex_lock"), dstring("pthread_mutex_lock_check"));
  symbol_deps.insert(lockcheck);

  std::pair<irep_idt, irep_idt> condcheck(
    dstring("pthread_cond_wait"), dstring("pthread_cond_wait_check"));
  symbol_deps.insert(condcheck);

  std::pair<irep_idt, irep_idt> joincheck(
    dstring("pthread_join"), dstring("pthread_join_noswitch"));
  symbol_deps.insert(joincheck);

  /* Iterate through the new_ctx symbols, figure out which ones to go into store_ctx
   *    For Python/Solidity this is everything: new_ctx already has a filtering layer
   *    For other frontends, only add symbols that exist already in context but value empty
   * store_ctx is what actually gets merged into the existing, final context
   */

  // Determine whether this language uses a whitelist-based loading strategy.
  // Python: uses whitelist with clib64 → symbols split between new_ctx/ignored_ctx.
  // Solidity: uses dedicated sol64 binary → ALL symbols in new_ctx, no whitelist.
  bool uses_whitelist = language && language->id() == "python";

  new_ctx.foreach_operand([&context,
                           &store_ctx,
                           &symbol_deps,
                           &to_include,
                           &is_solidity,
                           &uses_whitelist](const symbolt &s) {
    const symbolt *symbol = context.find_symbol(s.id);
    if (
      (is_solidity || uses_whitelist) ||
      (symbol != nullptr && symbol->value.is_nil()))
    {
      store_ctx.add(s);
      ingest_symbol(s.id, symbol_deps, to_include);
    }
  });

  /* Now iterate through the dependencies that we know we want to add (due to ingest_symbol filter)
   * These will be symbols that didn't make it into store_ctx
   *
   * For Python (whitelist):
   *    - symbols not in whitelist go to ignored_ctx, dependencies found there
   * For Solidity (dedicated binary, no whitelist):
   *    - all symbols already in new_ctx, dependencies found in new_ctx
   * For other frontends (no filter):
   *    - everything in new_ctx, dependencies found in new_ctx
   */
  for (std::list<irep_idt>::const_iterator nameit = to_include.begin();
       nameit != to_include.end();
       ++nameit)
  {
    symbolt *s;

    if (uses_whitelist)
    {
      s = ignored_ctx.find_symbol(*nameit);
    }
    else
    {
      s = new_ctx.find_symbol(*nameit);
    }

    if (s != nullptr)
    {
      store_ctx.add(*s);

      if (uses_whitelist)
      {
        generate_symbol_deps(s->id, s->value, symbol_deps);
        generate_symbol_deps(s->id, s->type, symbol_deps);
      }

      ingest_symbol(*nameit, symbol_deps, to_include);
    }
  }

  // Bring store_ctx symbols into context
  if (c_link(context, store_ctx, "<built-in-library>"))
  {
    // Merging failed
    log_error("Failed to merge C library");
    abort();
  }
  // We basically need a place where we know that ESBMC produces the "main" executable that will be run.
  // This is the best place that I've found and mimics how a real compiler would work:
  // First compile all source files to objects files, then link them together and then link with the libc
  // library. Only when linking to the libc library, we know that all unresolved extern symbols (those whose
  // value is nil) will stay unresolved. A normal linker would reject such files, but we provide some compatibility with
  // those and initialize the extern variables to nondet.
  context.Foreach_operand([&context](symbolt &s) {
    if (s.is_extern && !s.type.is_code())
    {
      log_debug(
        "c2goto",
        "extern variable with id {} not found, initializing value to "
        "nondet! "
        "This code would not compile with an actual compiler.",
        s.id);
      exprt value =
        exprt("sideeffect", get_complete_type(s.type, namespacet{context}));
      value.statement("nondet");
      s.value = value;
    }
  });
}
