/// \file solidity_convert_type.cpp
/// \brief Type conversion for the Solidity frontend.
///
/// Converts Solidity type descriptions (elementary types like uint256/bool/
/// address/string/bytes, array types, mapping types, struct types, enum
/// types, and contract types) from the solc JSON AST into ESBMC's irep2
/// type system (typet).

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <fstream>

bool solidity_convertert::get_type_description(
  const nlohmann::json &type_name,
  typet &new_type)
{
  return get_type_description(empty_json, type_name, new_type);
}

bool solidity_convertert::get_type_description(
  const nlohmann::json &decl,
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule type-name:
  SolidityGrammar::TypeNameT type = SolidityGrammar::get_type_name_t(type_name);

  std::string typeIdentifier;
  std::string typeString;

  if (type_name.contains("typeIdentifier"))
    typeIdentifier = type_name["typeIdentifier"].get<std::string>();
  if (type_name.contains("typeString"))
    typeString = type_name["typeString"].get<std::string>();

  log_debug(
    "solidity", "got type-name={}", SolidityGrammar::type_name_to_str(type));

  switch (type)
  {
  case SolidityGrammar::TypeNameT::ElementaryTypeName:
  case SolidityGrammar::TypeNameT::AddressTypeName:
  case SolidityGrammar::TypeNameT::AddressPayableTypeName:
  {
    // rule state-variable-declaration
    if (get_elementary_type_name(type_name, new_type))
      return true;
    break;
  }
  case SolidityGrammar::TypeNameT::ParameterList:
  {
    // rule parameter-list
    // Used for Solidity function parameter or return list
    if (get_parameter_list(type_name, new_type))
      return true;
    break;
  }
  case SolidityGrammar::TypeNameT::Pointer:
  {
    // FunctionTypeName parameter / struct field (internal or external
    // function types, e.g. `function(uint) pure returns (uint) f`).
    // Lowered to an opaque void* — indirect calls through it return
    // nondet values (handled in convert_call).
    if (
      type_name.contains("nodeType") &&
      type_name["nodeType"] == "FunctionTypeName")
    {
      new_type = gen_pointer_type(empty_typet());
      new_type.set("#sol_func_ptr", true);
      set_sol_type(new_type, SolidityGrammar::SolType::FUNC_PTR);
      break;
    }

    // typeDescriptions-only form (e.g. mapping value type, struct field
    // stored only as a typeIdentifier/typeString pair, or a builtin
    // function reference such as `abi.encode` captured as a tuple
    // component): any `t_function_*` kind is lowered to the same opaque
    // fn-ptr shape as the FunctionTypeName case above.
    if (typeIdentifier.compare(0, 11, "t_function_") == 0)
    {
      new_type = gen_pointer_type(empty_typet());
      new_type.set("#sol_func_ptr", true);
      set_sol_type(new_type, SolidityGrammar::SolType::FUNC_PTR);
      break;
    }

    // auxiliary type: pointer (FuncToPtr decay)
    // This part is for FunctionToPointer decay only
    assert(
      typeString.find("function") != std::string::npos ||
      typeString.find("contract") != std::string::npos);

    // Since Solidity does not have this, first make a pointee
    nlohmann::json pointee = make_pointee_type(type_name);
    typet sub_type;
    if (get_func_decl_ref_type(pointee, sub_type))
      return true;

    if (sub_type.is_struct() || sub_type.is_union())
    {
      log_error("struct or union pointer type is not supported");
      return true;
    }

    new_type = gen_pointer_type(sub_type);
    break;
  }
  case SolidityGrammar::TypeNameT::PointerArrayToPtr:
  {
    // auxiliary type: pointer (FuncToPtr decay)
    // This part is for FunctionToPointer decay only
    assert(typeIdentifier.find("ArrayToPtr") != std::string::npos);

    // Array type descriptor is like:
    //  "typeIdentifier": "ArrayToPtr",
    //  "typeString": "uint8[2] memory"

    // Since Solidity does not have this, first make a pointee
    typet sub_type;
    if (get_array_to_pointer_type(type_name, sub_type))
      return true;

    if (sub_type.is_struct() || sub_type.is_union())
    {
      log_error("struct or union pointer type is not supported");
      return true;
    }

    new_type = gen_pointer_type(sub_type);
    break;
  }
  case SolidityGrammar::TypeNameT::NestedArrayTypeName:
  {
    /* e.g.
    "typeDescriptions": {
        "typeIdentifier": "t_array$_t_array$_t_int256_$4_storage_$dyn_storage",
        "typeString": "int256[4][]"
    },
    "typeName": {
        "baseType": {
            "baseType": {
                "id": 2,
                "name": "int",
                "nodeType": "ElementaryTypeName",
                "typeDescriptions": {
                    "typeIdentifier": "t_int256",
                    "typeString": "int256"
                }
            },
            "id": 4,
            "length": {
                "hexValue": "34",
                "id": 3,
                "isConstant": false,
                "isLValue": false,
                "isPure": true,
                "kind": "number",
                "lValueRequested": false,
                "nodeType": "Literal",
                "typeDescriptions": {
                    "typeIdentifier": "t_rational_4_by_1",
                    "typeString": "int_const 4"
                },
                "value": "4"
            },
            "nodeType": "ArrayTypeName",
            "typeDescriptions": {
                "typeIdentifier": "t_array$_t_int256_$4_storage_ptr",
                "typeString": "int256[4]"
            }
        },
        "id": 5,
        "nodeType": "ArrayTypeName",
        "typeDescriptions": {
            "typeIdentifier": "t_array$_t_array$_t_int256_$4_storage_$dyn_storage_ptr",
            "typeString": "int256[4][]"
        }
    convert it to:

    pointer
    * subtype: array
        * size: constant
            * type: unsignedbv
                * width: 64
            * value: 0000000000000000000000000000000000000000000000000000000000000100
            * #cformat: 4
        * subtype: signedbv
            * width: 32
            * #cpp_type: signed_int
    */
    // B2: If every dimension is a compile-time fixed size, use a
    // native nested `array_typet(array_typet(T, inner_N), outer_M)`
    // embedded directly in the surrounding struct. This sidesteps the
    // `T**` + nested-calloc value-set aliasing bug entirely. Mixed
    // (at least one dynamic) shapes still fall through to the
    // pointer-backed path below.
    if (!decl.empty() && decl.contains("typeName"))
    {
      typet native_t;
      if (try_native_nested_fixed_array(decl["typeName"], native_t))
      {
        new_type = native_t;
        break;
      }
    }

    typet base_type;
    if (!decl.empty() && decl.contains("typeName") &&
        decl["typeName"].contains("baseType"))
    {
      // From variable declaration: use AST baseType node directly
      nlohmann::json inner_decl;
      inner_decl["typeName"] = decl["typeName"]["baseType"];
      if (get_type_description(
            inner_decl,
            decl["typeName"]["baseType"]["typeDescriptions"],
            base_type))
        return true;

      // Phase 5 Surface 1: preserve fixed-inner-array typing through outer
      // dyn-array wrap. The recursion above decays inner `T[N]` to
      // `pointer<T>` with `#sol_array_size=N` tag (via get_array_pointer_type).
      // If we hand that to get_array_pointer_type unchanged, the outer wrap
      // produces `pointer<pointer<T>>` (= T**) and the N tag is lost — which
      // creates a read/write asymmetry on `T[N][]` parameter access (writes
      // hit packed N*sizeof(T) byte offsets but reads follow T** typed
      // deref). Lift back to `array<T, N>` so the outer wrap produces
      // `pointer<array<T, N>>` and the per-row stride is preserved on both
      // sides. Gate strictly on pointer + #sol_array_size + SolType::ARRAY
      // (NOT DYNARRAY) so fully-dyn `T[][]` and 1D dyn paths are unaffected.
      if (
        base_type.id() == "pointer" &&
        !base_type.get("#sol_array_size").empty() &&
        get_sol_type(base_type) == SolidityGrammar::SolType::ARRAY)
      {
        const std::string n_str = id2string(base_type.get("#sol_array_size"));
        try
        {
          unsigned long n = std::stoul(n_str);
          if (n != 0)
          {
            constant_exprt sz(
              integer2binary(BigInt(n), bv_width(int_type())),
              integer2string(BigInt(n)),
              int_type());
            typet inner_subtype = base_type.subtype();
            base_type = array_typet(inner_subtype, sz);
            // Preserve the size tag at the new array_typet level so
            // downstream code that reads #sol_array_size still works.
            base_type.set("#sol_array_size", n_str);
            set_sol_type(base_type, SolidityGrammar::SolType::ARRAY);
          }
        }
        catch (const std::exception &)
        {
          // base_type stays as pointer<T> (degraded); outer wrap produces
          // T** as before. Soundness preserved by falling through.
        }
      }

      if (get_array_pointer_type(decl, base_type, new_type))
        return true;
    }
    else
    {
      // From expression context (no decl): extract base type from strings.
      // e.g. typeIdentifier "t_array$_t_array$_t_uint256_$dyn_storage_$dyn_storage"
      //      typeString     "uint256[] storage ref[] storage ref"
      // Base element is "uint256[]" / "t_array$_t_uint256_$dyn_storage"
      //
      // Also handles fixed outer arrays:
      // e.g. typeIdentifier "t_array$_t_array$_t_uint256_$dyn_storage_$3_storage"
      //      typeString     "uint256[] storage ref[3] storage ref"
      // Base element is "uint256[]" / "t_array$_t_uint256_$dyn_storage"
      const std::string prefix = "t_array$_";
      std::string rest = typeIdentifier.substr(prefix.size());

      // Find the outer array's delimiter: the last "_$" followed by
      // "dyn" or digits, then "_storage"/"_memory"/etc.
      // This correctly skips inner type delimiters.
      // e.g. rest = "t_array$_t_uint256_$dyn_storage_$3_storage"
      //   → find last "_$" at the "_$3" position, outer_size = "3"
      // e.g. rest = "t_array$_t_uint256_$dyn_storage_$dyn_storage"
      //   → find last "_$" at the "_$dyn" (outer), outer is dynamic
      std::string base_id;
      bool outer_is_dynamic = true;
      std::string outer_size_str;
      // Scan backwards for the last "_$" that starts the outer delimiter
      size_t last_delim = rest.rfind("_$");
      if (last_delim != std::string::npos)
      {
        std::string after_delim = rest.substr(last_delim + 2);
        if (after_delim.compare(0, 3, "dyn") == 0)
        {
          base_id = rest.substr(0, last_delim);
          outer_is_dynamic = true;
        }
        else if (!after_delim.empty() && std::isdigit(after_delim[0]))
        {
          // Fixed-size: extract digits
          size_t digit_end = 0;
          while (digit_end < after_delim.size() &&
                 std::isdigit(after_delim[digit_end]))
            digit_end++;
          outer_size_str = after_delim.substr(0, digit_end);
          base_id = rest.substr(0, last_delim);
          outer_is_dynamic = false;
        }
        else
        {
          // Not a valid delimiter; try second-to-last "_$"
          size_t prev_delim = rest.rfind("_$", last_delim - 1);
          if (prev_delim != std::string::npos)
          {
            std::string after = rest.substr(prev_delim + 2);
            if (after.compare(0, 3, "dyn") == 0)
            {
              base_id = rest.substr(0, prev_delim);
              outer_is_dynamic = true;
            }
            else
            {
              base_id = rest.substr(0, prev_delim);
            }
          }
          else
          {
            base_id = rest;
          }
        }
      }
      else
      {
        base_id = rest;
      }

      // Extract base typeString: strip trailing "[<size>]" from typeString
      std::string base_ts = typeString;
      auto strip_loc = [](std::string &s) {
        for (const char *suf :
             {" storage ref", " storage", " memory", " calldata"})
        {
          if (s.size() > strlen(suf) &&
              s.compare(s.size() - strlen(suf), strlen(suf), suf) == 0)
          {
            s.erase(s.size() - strlen(suf));
            return;
          }
        }
      };
      strip_loc(base_ts);
      // Remove trailing "[<optional_size>]"
      auto last_bracket = base_ts.rfind('[');
      if (last_bracket != std::string::npos)
        base_ts.erase(last_bracket);
      strip_loc(base_ts); // strip again for inner location qualifier

      nlohmann::json base_json;
      base_json["typeIdentifier"] = base_id;
      base_json["typeString"] = base_ts;
      if (get_type_description(base_json, base_type))
        return true;

      if (outer_is_dynamic)
      {
        new_type = gen_pointer_type(base_type);
        set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
      }
      else
      {
        // Pathological huge sizes (e.g. `T[2**240]`) overflow `unsigned
        // long` and crash stoul. We cannot materialise that many elements
        // anyway, so degrade to DYNARRAY (pointer model) and let downstream
        // accesses lower to nondet reads. Sound over-approximation.
        unsigned long z_ext_value = 0;
        try
        {
          z_ext_value = std::stoul(outer_size_str, nullptr);
        }
        catch (const std::exception &)
        {
          new_type = gen_pointer_type(base_type);
          set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
          break;
        }
        new_type = array_typet(
          base_type,
          constant_exprt(
            integer2binary(z_ext_value, bv_width(int_type())),
            integer2string(z_ext_value),
            int_type()));
        new_type.set("#sol_array_size", outer_size_str);
        set_sol_type(new_type, SolidityGrammar::SolType::ARRAY);
      }
    }

    break;
  }
  case SolidityGrammar::TypeNameT::ArrayTypeName:
  case SolidityGrammar::TypeNameT::DynArrayTypeName:
  {
    // Deal with array with constant size, e.g., int a[2]; Similar to clang::Type::ConstantArray
    // array's typeDescription is in a compact form, e.g.:
    //    "typeIdentifier": "t_array$_t_uint8_$2_storage_ptr",
    //    "typeString": "uint8[2]"
    // We need to extract the elementary type of array from the information provided above
    // We want to make it like ["baseType"]["typeDescriptions"]

    typet the_type;
    exprt the_size;
    if (!decl.empty())
    {
      // access from get_var_decl
      assert(decl["typeName"].contains("baseType"));
      if (get_type_description(
            decl["typeName"]["baseType"]["typeDescriptions"], the_type))
        return true;

      if (get_array_pointer_type(decl, the_type, new_type))
        return true;
    }
    else if (type == SolidityGrammar::TypeNameT::ArrayTypeName)
    {
      // for tuple array
      nlohmann::json array_elementary_type =
        make_array_elementary_type(type_name);

      if (get_type_description(array_elementary_type, the_type))
        return true;

      std::string the_size = get_array_size(type_name);
      unsigned z_ext_value = 0;
      try
      {
        z_ext_value = std::stoul(the_size, nullptr);
      }
      catch (const std::exception &)
      {
        // Size does not fit in unsigned long (e.g. uint[2**100]).
        // Degrade to a dynamic-array pointer model so we don't crash.
        new_type = gen_pointer_type(the_type);
        set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
        break;
      }
      new_type = array_typet(
        the_type,
        constant_exprt(
          integer2binary(z_ext_value, bv_width(int_type())),
          integer2string(z_ext_value),
          int_type()));
      new_type.set("#sol_array_size", the_size);
      set_sol_type(new_type, SolidityGrammar::SolType::ARRAY_LITERAL);
    }
    else
    {
      // e.g.
      // "typeDescriptions": {
      //     "typeIdentifier": "t_array$_t_uint256_$dyn_memory_ptr",
      //     "typeString": "uint256[]"

      // 1. rebuild baseType
      nlohmann::json new_json;
      std::string temp = typeString;
      auto pos = temp.find("[]"); // e.g. "uint256[] memory"
      const std::string new_typeString = temp.substr(0, pos);

      // Extract element type identifier from array type identifier.
      // e.g. "t_array$_t_uint256_$dyn_memory_ptr" => "t_uint256"
      //      "t_array$_t_struct$_Message_$11_storage_$dyn_storage"
      //        => "t_struct$_Message_$11_storage"
      //      "t_array$_t_mapping$_t_uint256_$_t_uint256_$_$dyn_storage"
      //        => "t_mapping$_t_uint256_$_t_uint256_$"
      auto extract = [](const std::string &s) -> std::string {
        // strip "t_array$_" prefix
        const std::string prefix = "t_array$_";
        if (s.compare(0, prefix.size(), prefix) != 0)
          return "";
        std::string rest = s.substr(prefix.size());
        // find "_$dyn" suffix and remove it (dynamic arrays)
        size_t dyn = rest.find("_$dyn");
        if (dyn != std::string::npos)
          return rest.substr(0, dyn);
        // fixed-size array: find "_$<digits>_" suffix
        // scan backwards from end for "_$" followed by digits
        for (size_t i = rest.size(); i >= 2; --i)
        {
          if (rest[i - 2] == '_' && rest[i - 1] == '$' && i < rest.size() &&
              std::isdigit(rest[i]))
            return rest.substr(0, i - 2);
        }
        return "";
      };
      const std::string new_typeIdentifier = extract(typeIdentifier);
      log_debug("solidity", "new_typeIdentifier = {}", new_typeIdentifier);
      new_json["typeString"] = new_typeString;
      new_json["typeIdentifier"] = new_typeIdentifier;

      // 2. get subType
      typet sub_type;
      if (get_type_description(new_json, sub_type))
        return true;

      // 3. For mapping element types, model as 2D infinite array instead of
      //    pointer.  mapping(K=>V)[] is semantically equivalent to
      //    mapping(uint => mapping(K=>V)) + a length counter.  The pointer/
      //    malloc model cannot handle infinite-sized mapping elements.
      if (get_sol_type(sub_type) == SolidityGrammar::SolType::MAPPING &&
          sub_type.is_array())
      {
        new_type = array_typet();
        new_type.size(exprt("infinity"));
        new_type.subtype() = sub_type;
        new_type.set("#sol_mapping_array", true);
        set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
      }
      else
      {
        new_type = gen_pointer_type(sub_type);
        set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
      }
    }

    break;
  }
  case SolidityGrammar::TypeNameT::ContractTypeName:
  {
    // e.g. ContractName tmp = new ContractName(Args);
    // typeString is normally "contract ContractName" but `super`
    // inside a derived contract arrives here as
    // "contract super ContractName" (typeIdentifier `t_super$_X_$N`).
    // Take the last whitespace-separated token as the contract name.

    std::string constructor_name = typeString;
    size_t pos = constructor_name.rfind(" ");
    std::string cname = constructor_name.substr(pos + 1);
    std::string id = prefix + cname;

    new_type = pointer_typet(symbol_typet(id));
    set_sol_type(new_type, SolidityGrammar::SolType::CONTRACT);
    new_type.set("#sol_contract", cname);
    break;
  }
  case SolidityGrammar::TypeNameT::TypeConversionName:
  {
    // e.g.
    // uint32 a = 0x432178;
    // uint16 b = uint16(a); // b will be 0x2178 now
    // "nodeType": "TypeConversionExpression",
    //             "src": "155:6:0",
    //             "typeDescriptions": {
    //                 "typeIdentifier": "t_type$_t_uint16_$",
    //                 "typeString": "type(uint16)"
    //             },
    //             "typeName": {
    //                 "id": 10,
    //                 "name": "uint16",
    //                 "nodeType": "ElementaryTypeName",
    //                 "src": "155:6:0",
    //                 "typeDescriptions": {}
    //             }

    nlohmann::json new_json;

    // convert it back to ElementaryTypeName by removing the "type" prefix
    std::size_t begin = typeIdentifier.find("$_");
    std::size_t end = typeIdentifier.rfind("_$");
    std::string new_typeIdentifier =
      typeIdentifier.substr(begin + 2, end - begin - 2);

    begin = typeString.find("type(");
    end = typeString.rfind(")");
    std::string new_typeString = typeString.substr(begin + 5, end - begin - 5);

    new_json["typeIdentifier"] = new_typeIdentifier;
    new_json["typeString"] = new_typeString;

    if (get_type_description(new_json, new_type))
      return true;

    break;
  }
  case SolidityGrammar::TypeNameT::EnumTypeName:
  {
    new_type = enum_type();
    set_sol_type(new_type, SolidityGrammar::SolType::ENUM);
    break;
  }
  case SolidityGrammar::TypeNameT::StructTypeName:
  {
    // e.g. struct ContractName.StructName
    //   "typeDescriptions": {
    //   "typeIdentifier": "t_struct$_Book_$8_storage",
    //   "typeString": "struct Base.Book storage ref"
    // }

    // extract id and ref_id;
    std::string delimiter = " ";

    int cnt = 1;
    std::string token;
    std::string _typeString = typeString;

    // extract the second string
    while (cnt >= 0)
    {
      if (_typeString.find(delimiter) == std::string::npos)
      {
        token = _typeString;
        break;
      }
      size_t pos = _typeString.find(delimiter);
      token = _typeString.substr(0, pos);
      _typeString.erase(0, pos + delimiter.length());
      cnt--;
    }

    const std::string id = prefix + "struct " + token;
    new_type = symbol_typet(id);
    set_sol_type(new_type, SolidityGrammar::SolType::STRUCT);
    break;
  }
  case SolidityGrammar::TypeNameT::MappingTypeName:
  {
    /*
        "typeIdentifier": "t_mapping$_t_address_$_t_uint256_$",
        "typeString": "mapping(address => uint256)"
    */
    // Mappings declared inside free-standing structs (e.g. `struct itmap {
    // mapping(uint => IndexValue) data; ... }` declared at file scope) are
    // converted before any contract scope is set. There's no contract to
    // gate `is_new_expr` against — fall through to the slow infinite-array
    // encoding which is correct in every mode (`should_treat_as_new("")`
    // returns false, but explicit short-circuit avoids relying on that).
    bool is_new_expr = !current_baseContractName.empty() &&
                       should_treat_as_new(current_baseContractName);

    if (is_new_expr)
      new_type = symbol_typet(lib_prefix + "mapping_t");
    else
    {
      // we will populate the size type later
      new_type = array_typet();
      new_type.size(exprt("infinity"));
      // Wide-BV index: mapping keys are up to 256 bits in Solidity; we
      // index the per-mapping infinite array directly with the raw key
      // (no XOR fold). The index_width annotation flows through migrate
      // to array_type2t::index_width and is consumed by smt_conv +
      // adjust_index. Closes ledger #22's path-1 256→64 fold gap.
      new_type.set("#esbmc_index_width", "256");
    }
    set_sol_type(new_type, SolidityGrammar::SolType::MAPPING);
    break;
  }
  case SolidityGrammar::TypeNameT::TupleTypeName:
  {
    // do nothing as it won't be used
    new_type = struct_typet();
    new_type.set("#cpp_type", "void");
    set_sol_type(new_type, SolidityGrammar::SolType::TUPLE_RETURNS);
    break;
  }
  case SolidityGrammar::TypeNameT::ErrorTypeName:
  {
    new_type = empty_typet();
    new_type.set("#cpp_type", "void");
    break;
  }
  case SolidityGrammar::TypeNameT::UserDefinedTypeName:
  {
    new_type = UserDefinedVarMap[typeString];
    break;
  }
  default:
  {
    log_error(
      "Unimplemented type in rule type-name: {}",
      SolidityGrammar::type_name_to_str(type));
    return true;
  }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict

  // set data location
  if (typeIdentifier.find("_memory_ptr") != std::string::npos)
    new_type.set("#sol_data_loc", "memory");
  else if (typeIdentifier.find("_storage_ptr") != std::string::npos)
    new_type.set("#sol_data_loc", "storage");
  else if (typeIdentifier.find("_calldata_ptr") != std::string::npos)
    new_type.set("#sol_data_loc", "calldata");

  return false;
}

bool solidity_convertert::get_func_decl_ref_type(
  const nlohmann::json &decl,
  typet &new_type)
{
  // For FunctionToPointer decay:
  // Get type when we make a function call:
  //  - FunnctionNoProto: x = nondet()
  //  - FunctionProto:    z = add(x, y)
  // Similar to the function get_type_description()
  SolidityGrammar::FunctionDeclRefT type =
    SolidityGrammar::get_func_decl_ref_t(decl);

  log_debug(
    "solidity",
    "\t@@@ got SolidityGrammar::FunctionDeclRefT = {}",
    SolidityGrammar::func_decl_ref_to_str(type));

  switch (type)
  {
  case SolidityGrammar::FunctionDeclRefT::FunctionNoProto:
  {
    code_typet type;
    // Return type
    const nlohmann::json &rtn_type = decl["returnParameters"];

    typet return_type;
    if (get_type_description(rtn_type, return_type))
      return true;

    type.return_type() = return_type;

    if (!type.arguments().size())
      type.make_ellipsis();

    new_type = type;
    break;
  }
  case SolidityGrammar::FunctionDeclRefT::FunctionProto:
  {
    code_typet type;

    // store current state
    const nlohmann::json *old_functionDecl = current_functionDecl;
    const std::string old_functionName = current_functionName;

    // need in get_function_params()
    assert(decl.contains("name"));
    current_functionName = decl["name"].get<std::string>();
    current_functionDecl = &decl;

    std::string current_contractName;
    get_current_contract_name(decl, current_contractName);

    if (decl.contains("returnParameters"))
    {
      const nlohmann::json &rtn_type = decl["returnParameters"];

      typet return_type;
      if (get_type_description(rtn_type, return_type))
        return true;

      type.return_type() = return_type;
    }

    // convert parameters if the function has them
    // update the typet, since typet contains parameter annotations
    for (const auto &decl : decl["parameters"]["parameters"].items())
    {
      const nlohmann::json &func_param_decl = decl.value();

      code_typet::argumentt param;
      if (get_function_params(func_param_decl, current_contractName, param))
        return true;

      type.arguments().push_back(param);
    }

    current_functionName = old_functionName;
    current_functionDecl = old_functionDecl;

    new_type = type;
    break;
  }
  default:
  {
    log_debug(
      "solidity",
      "	@@@ Got type={}",
      SolidityGrammar::func_decl_ref_to_str(type));
    return true;
  }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict
  return false;
}

bool solidity_convertert::get_array_to_pointer_type(
  const nlohmann::json &type_descriptor,
  typet &new_type)
{
  // Function to get the base type in ArrayToPointer decay.
  // Recognise the common element-type prefixes in Solidity's
  // typeString (e.g. "uint8[]", "uint256[] calldata",
  // "int256[]", "address[]") and map to the matching scalar type.
  const std::string ts =
    type_descriptor["typeString"].get<std::string>();

  // uintN / intN
  auto try_int_width = [&](const std::string &prefix, bool is_signed) -> bool {
    auto pos = ts.find(prefix);
    if (pos == std::string::npos)
      return false;
    size_t e = pos + prefix.size();
    size_t start = e;
    while (e < ts.size() && std::isdigit(static_cast<unsigned char>(ts[e])))
      ++e;
    if (e == start)
      return false;
    unsigned bits = std::stoul(ts.substr(start, e - start));
    if (bits == 0 || bits > 256 || (bits % 8) != 0)
      return false;
    new_type = is_signed ? (typet)signedbv_typet(bits)
                         : (typet)unsignedbv_typet(bits);
    new_type.set("#cpp_type", is_signed ? "signed_int" : "unsigned_int");
    return true;
  };

  if (try_int_width("uint", false))
    return false;
  if (try_int_width("int", true))
    return false;
  if (ts.find("address") != std::string::npos)
  {
    new_type = unsignedbv_typet(160);
    new_type.set("#cpp_type", "unsigned_int");
    return false;
  }
  if (ts.find("bool") != std::string::npos)
  {
    new_type = bool_typet();
    return false;
  }

  log_error("Unsupported types in ArrayToPointer decay: {}", ts);
  return true;
}

// parse a tuple to struct

bool solidity_convertert::get_elementary_type_name_uint(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int uint_size = SolidityGrammar::uint_type_name_to_size(type);
  out = unsignedbv_typet(uint_size);

  return false;
}

/**
     * @brief Populate the out `typet` parameter with the int type specified by type parameter
     *
     * @param type The type of the int to be poulated
     * @param out The variable that holds the resulting type
     * @return false iff population was successful
     */
bool solidity_convertert::get_elementary_type_name_int(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  const unsigned int int_size = SolidityGrammar::int_type_name_to_size(type);
  out = signedbv_typet(int_size);

  return false;
}

bool solidity_convertert::get_elementary_type_name_bytesn(
  SolidityGrammar::ElementaryTypeNameT &type,
  typet &out)
{
  /*
    bytes1 has size of 8 bits (possible values 0x00 to 0xff),
    which you can implicitly convert to uint8 (unsigned integer of size 8 bits) but not to int8
  */
  const unsigned int byte_num = SolidityGrammar::bytesn_type_name_to_size(type);
  out = unsignedbv_typet(byte_num * 8);

  return false;
}

bool solidity_convertert::get_elementary_type_name(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule elementary-type-name:
  // equivalent to clang's get_builtin_type()
  SolidityGrammar::ElementaryTypeNameT type =
    SolidityGrammar::get_elementary_type_name_t(type_name);

  log_debug(
    "solidity",
    "	@@@ got ElementaryType: SolidityGrammar::ElementaryTypeNameT::{}",
    fmt::underlying(type));

  switch (type)
  {
  // rule unsigned-integer-type
  case SolidityGrammar::ElementaryTypeNameT::UINT8:
  case SolidityGrammar::ElementaryTypeNameT::UINT16:
  case SolidityGrammar::ElementaryTypeNameT::UINT24:
  case SolidityGrammar::ElementaryTypeNameT::UINT32:
  case SolidityGrammar::ElementaryTypeNameT::UINT40:
  case SolidityGrammar::ElementaryTypeNameT::UINT48:
  case SolidityGrammar::ElementaryTypeNameT::UINT56:
  case SolidityGrammar::ElementaryTypeNameT::UINT64:
  case SolidityGrammar::ElementaryTypeNameT::UINT72:
  case SolidityGrammar::ElementaryTypeNameT::UINT80:
  case SolidityGrammar::ElementaryTypeNameT::UINT88:
  case SolidityGrammar::ElementaryTypeNameT::UINT96:
  case SolidityGrammar::ElementaryTypeNameT::UINT104:
  case SolidityGrammar::ElementaryTypeNameT::UINT112:
  case SolidityGrammar::ElementaryTypeNameT::UINT120:
  case SolidityGrammar::ElementaryTypeNameT::UINT128:
  case SolidityGrammar::ElementaryTypeNameT::UINT136:
  case SolidityGrammar::ElementaryTypeNameT::UINT144:
  case SolidityGrammar::ElementaryTypeNameT::UINT152:
  case SolidityGrammar::ElementaryTypeNameT::UINT160:
  case SolidityGrammar::ElementaryTypeNameT::UINT168:
  case SolidityGrammar::ElementaryTypeNameT::UINT176:
  case SolidityGrammar::ElementaryTypeNameT::UINT184:
  case SolidityGrammar::ElementaryTypeNameT::UINT192:
  case SolidityGrammar::ElementaryTypeNameT::UINT200:
  case SolidityGrammar::ElementaryTypeNameT::UINT208:
  case SolidityGrammar::ElementaryTypeNameT::UINT216:
  case SolidityGrammar::ElementaryTypeNameT::UINT224:
  case SolidityGrammar::ElementaryTypeNameT::UINT232:
  case SolidityGrammar::ElementaryTypeNameT::UINT240:
  case SolidityGrammar::ElementaryTypeNameT::UINT248:
  case SolidityGrammar::ElementaryTypeNameT::UINT256:
  {
    if (get_elementary_type_name_uint(type, new_type))
      return true;

    set_sol_type(new_type, SolidityGrammar::elementary_to_sol_type(type));
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::INT8:
  case SolidityGrammar::ElementaryTypeNameT::INT16:
  case SolidityGrammar::ElementaryTypeNameT::INT24:
  case SolidityGrammar::ElementaryTypeNameT::INT32:
  case SolidityGrammar::ElementaryTypeNameT::INT40:
  case SolidityGrammar::ElementaryTypeNameT::INT48:
  case SolidityGrammar::ElementaryTypeNameT::INT56:
  case SolidityGrammar::ElementaryTypeNameT::INT64:
  case SolidityGrammar::ElementaryTypeNameT::INT72:
  case SolidityGrammar::ElementaryTypeNameT::INT80:
  case SolidityGrammar::ElementaryTypeNameT::INT88:
  case SolidityGrammar::ElementaryTypeNameT::INT96:
  case SolidityGrammar::ElementaryTypeNameT::INT104:
  case SolidityGrammar::ElementaryTypeNameT::INT112:
  case SolidityGrammar::ElementaryTypeNameT::INT120:
  case SolidityGrammar::ElementaryTypeNameT::INT128:
  case SolidityGrammar::ElementaryTypeNameT::INT136:
  case SolidityGrammar::ElementaryTypeNameT::INT144:
  case SolidityGrammar::ElementaryTypeNameT::INT152:
  case SolidityGrammar::ElementaryTypeNameT::INT160:
  case SolidityGrammar::ElementaryTypeNameT::INT168:
  case SolidityGrammar::ElementaryTypeNameT::INT176:
  case SolidityGrammar::ElementaryTypeNameT::INT184:
  case SolidityGrammar::ElementaryTypeNameT::INT192:
  case SolidityGrammar::ElementaryTypeNameT::INT200:
  case SolidityGrammar::ElementaryTypeNameT::INT208:
  case SolidityGrammar::ElementaryTypeNameT::INT216:
  case SolidityGrammar::ElementaryTypeNameT::INT224:
  case SolidityGrammar::ElementaryTypeNameT::INT232:
  case SolidityGrammar::ElementaryTypeNameT::INT240:
  case SolidityGrammar::ElementaryTypeNameT::INT248:
  case SolidityGrammar::ElementaryTypeNameT::INT256:
  {
    if (get_elementary_type_name_int(type, new_type))
      return true;

    set_sol_type(new_type, SolidityGrammar::elementary_to_sol_type(type));
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::INT_LITERAL:
  {
    // for int_const type
    new_type = signedbv_typet(256);
    new_type.set("#cpp_type", "signed_char");
    set_sol_type(new_type, SolidityGrammar::SolType::INT_CONST);
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BOOL:
  {
    new_type = bool_t;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::STRING:
  {
    // cpp: std::string str;
    // new_type = symbol_typet("tag-std::string");
    new_type = string_t;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS:
  {
    //  An Address is a DataHexString of 20 bytes (uint160)
    // e.g. 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984
    // ops: <=, <, ==, !=, >= and >
    new_type = addr_t;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::ADDRESS_PAYABLE:
  {
    new_type = addrp_t;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BYTES1:
  case SolidityGrammar::ElementaryTypeNameT::BYTES2:
  case SolidityGrammar::ElementaryTypeNameT::BYTES3:
  case SolidityGrammar::ElementaryTypeNameT::BYTES4:
  case SolidityGrammar::ElementaryTypeNameT::BYTES5:
  case SolidityGrammar::ElementaryTypeNameT::BYTES6:
  case SolidityGrammar::ElementaryTypeNameT::BYTES7:
  case SolidityGrammar::ElementaryTypeNameT::BYTES8:
  case SolidityGrammar::ElementaryTypeNameT::BYTES9:
  case SolidityGrammar::ElementaryTypeNameT::BYTES10:
  case SolidityGrammar::ElementaryTypeNameT::BYTES11:
  case SolidityGrammar::ElementaryTypeNameT::BYTES12:
  case SolidityGrammar::ElementaryTypeNameT::BYTES13:
  case SolidityGrammar::ElementaryTypeNameT::BYTES14:
  case SolidityGrammar::ElementaryTypeNameT::BYTES15:
  case SolidityGrammar::ElementaryTypeNameT::BYTES16:
  case SolidityGrammar::ElementaryTypeNameT::BYTES17:
  case SolidityGrammar::ElementaryTypeNameT::BYTES18:
  case SolidityGrammar::ElementaryTypeNameT::BYTES19:
  case SolidityGrammar::ElementaryTypeNameT::BYTES20:
  case SolidityGrammar::ElementaryTypeNameT::BYTES21:
  case SolidityGrammar::ElementaryTypeNameT::BYTES22:
  case SolidityGrammar::ElementaryTypeNameT::BYTES23:
  case SolidityGrammar::ElementaryTypeNameT::BYTES24:
  case SolidityGrammar::ElementaryTypeNameT::BYTES25:
  case SolidityGrammar::ElementaryTypeNameT::BYTES26:
  case SolidityGrammar::ElementaryTypeNameT::BYTES27:
  case SolidityGrammar::ElementaryTypeNameT::BYTES28:
  case SolidityGrammar::ElementaryTypeNameT::BYTES29:
  case SolidityGrammar::ElementaryTypeNameT::BYTES30:
  case SolidityGrammar::ElementaryTypeNameT::BYTES31:
  case SolidityGrammar::ElementaryTypeNameT::BYTES32:
  {
    new_type = byte_static_t;
    new_type.set("#sol_bytesn_size", bytesn_type_name_to_size(type));
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::BYTES:
  {
    new_type = byte_dynamic_t;
    break;
  }
  case SolidityGrammar::ElementaryTypeNameT::STRING_LITERAL:
  {
    // Try to locate the owning Literal node to recover the exact
    // char-array type. find_last_parent uses deep equality, so if the
    // same typeDescriptions appears under an arguments[] array the
    // returned parent can itself be that array (no "value" key).
    // Fall back to the generic string_t (char*) in that case.
    const auto &json = find_last_parent(src_ast_json, type_name);
    if (json.is_object() && json.contains("value") && json["value"].is_string())
    {
      string_constantt x(json["value"].get<std::string>());
      set_sol_type(x.type(), SolidityGrammar::SolType::STRING_LITERAL);
      new_type = x.type();
    }
    else
    {
      new_type = string_t;
      set_sol_type(new_type, SolidityGrammar::SolType::STRING_LITERAL);
    }

    break;
  }
  default:
  {
    log_debug(
      "solidity",
      "	@@@ Got elementary-type-name={}",
      SolidityGrammar::elementary_type_name_to_str(type));
    log_error(
      "Unimplemented type in rule elementary-type-name: {}",
      SolidityGrammar::elementary_type_name_to_str(type));
    return true;
  }
  }

  //TODO set #extint
  // switch (type)
  // {
  // case SolidityGrammar::ElementaryTypeNameT::BOOL:
  // case SolidityGrammar::ElementaryTypeNameT::STRING:
  // {
  //   break;
  // }
  // default:
  // {
  //   new_type.set("#extint", true);
  //   break;
  // }
  // }

  return false;
}

bool solidity_convertert::get_parameter_list(
  const nlohmann::json &type_name,
  typet &new_type)
{
  // For Solidity rule parameter-list:
  //  - For non-empty param list, it may need to call get_elementary_type_name, since parameter-list is just a list of types
  SolidityGrammar::ParameterListT type =
    SolidityGrammar::get_parameter_list_t(type_name);

  log_debug(
    "solidity",
    "\tGot ParameterList {}",
    SolidityGrammar::parameter_list_to_str(type));

  switch (type)
  {
  case SolidityGrammar::ParameterListT::EMPTY:
  {
    // equivalent to clang's "void"
    new_type = empty_typet();
    new_type.set("#cpp_type", "void");
    break;
  }
  case SolidityGrammar::ParameterListT::ONE_PARAM:
  {
    assert(type_name["parameters"].size() == 1);

    const nlohmann::json &rtn_type = type_name["parameters"].at(0);
    if (rtn_type.contains("typeName"))
    {
      if (get_type_description(
            rtn_type, rtn_type["typeName"]["typeDescriptions"], new_type))
        return true;
    }
    else
    {
      if (get_type_description(rtn_type["typeDescriptions"], new_type))
        return true;
    }

    break;
  }
  case SolidityGrammar::ParameterListT::MORE_THAN_ONE_PARAM:
  {
    // if contains multiple return types
    // We will return null because we create the symbols of the struct accordingly
    assert(type_name["parameters"].size() > 1);
    new_type = empty_typet();
    new_type.set("#cpp_type", "void");
    set_sol_type(new_type, SolidityGrammar::SolType::TUPLE_RETURNS);
    break;
  }
  default:
  {
    log_error("Unimplemented type in rule parameter-list");
    return true;
  }
  }

  return false;
}

// parse the state variable

bool solidity_convertert::try_native_nested_fixed_array(
  const nlohmann::json &type_name_node,
  typet &result)
{
  // Walk the AST typeName chain from outermost to innermost, collecting
  // each ArrayTypeName level's `length` literal. Bail out to the
  // pointer-backed path if any level is dynamic or the length literal
  // cannot be decoded.
  std::vector<unsigned long long> dims;
  const nlohmann::json *cur = &type_name_node;

  while (cur->contains("baseType"))
  {
    if (!cur->contains("length") || (*cur)["length"].is_null())
      return false;
    const auto &len_node = (*cur)["length"];
    std::string len_str;
    if (
      len_node.is_object() && len_node.contains("value") &&
      len_node["value"].is_string())
      len_str = len_node["value"].get<std::string>();
    if (len_str.empty())
      return false;
    unsigned long long n = 0;
    try
    {
      n = std::stoull(len_str);
    }
    catch (const std::exception &)
    {
      return false;
    }
    dims.push_back(n);
    cur = &(*cur)["baseType"];
  }

  // Need at least two dimensions to enter the native-nested path.
  // 1D fixed arrays stay on the pointer model for now.
  if (dims.size() < 2)
    return false;

  if (!cur->contains("typeDescriptions"))
    return false;

  typet leaf_t;
  if (get_type_description((*cur)["typeDescriptions"], leaf_t))
    return false;

  // Build nested array_typet from innermost out.
  typet acc = leaf_t;
  for (auto it = dims.rbegin(); it != dims.rend(); ++it)
  {
    constant_exprt sz(
      integer2binary(BigInt(*it), bv_width(int_type())),
      integer2string(BigInt(*it)),
      int_type());
    acc = array_typet(acc, sz);
  }

  // Tag the outermost dim so downstream sites that expect
  // #sol_array_size on the outer fixed-array wrapper still see it.
  acc.set("#sol_array_size", std::to_string(dims.front()));
  set_sol_type(acc, SolidityGrammar::SolType::ARRAY);
  result = acc;
  return true;
}

bool solidity_convertert::get_array_pointer_type(
  const nlohmann::json &decl,
  const typet &base_type,
  typet &new_type)
{
  // For arrays whose element is a mapping, model the WHOLE array as an
  // infinite-sized 2D array instead of a pointer or fixed-size struct
  // member.  `mapping(K=>V)[N]` is semantically a finite map from
  // [0..N) to mapping(K=>V); the mapping element is itself unbounded,
  // so the pointer/malloc model cannot allocate it (and trying to lay
  // it out as a fixed-size struct member trips
  // array_type2t::get_width on the inner mapping during state-var
  // initialisation). The same treatment already applies to dynamic
  // arrays of mappings.
  if (
    get_sol_type(base_type) == SolidityGrammar::SolType::MAPPING &&
    base_type.is_array())
  {
    new_type = array_typet();
    new_type.size(exprt("infinity"));
    new_type.subtype() = base_type;
    new_type.set("#sol_mapping_array", true);
    set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
    return false;
  }

  new_type = gen_pointer_type(base_type);
  if (decl["typeName"].contains("length"))
  {
    std::string length;
    const auto &len_node = decl["typeName"]["length"];
    if (len_node.contains("value") && len_node["value"].is_string())
      length = len_node["value"].get<std::string>();

    // Prefer the array type's own typeString first: solc has already
    // constant-folded the length (e.g. "uint256[10]"), so this is the
    // authoritative post-folding size and avoids chasing constant refs
    // through TupleExpression/BinaryOperation value nodes that
    // get_constant_value cannot handle.
    if (length.empty())
    {
      const std::string ts =
        decl["typeName"].contains("typeDescriptions") &&
            decl["typeName"]["typeDescriptions"].contains("typeString") &&
            decl["typeName"]["typeDescriptions"]["typeString"].is_string()
          ? decl["typeName"]["typeDescriptions"]["typeString"].get<std::string>()
          : std::string();
      auto lb = ts.rfind('[');
      auto rb = ts.rfind(']');
      if (lb != std::string::npos && rb != std::string::npos && rb > lb + 1)
      {
        const std::string inner = ts.substr(lb + 1, rb - lb - 1);
        if (!inner.empty() &&
            std::all_of(inner.begin(), inner.end(), [](unsigned char ch) {
              return std::isdigit(ch);
            }))
          length = inner;
      }
    }

    // Last resort: follow the constant reference (simple literal chains).
    if (
      length.empty() && len_node.contains("referencedDeclaration") &&
      len_node["referencedDeclaration"].is_number_integer() &&
      len_node["referencedDeclaration"].get<int>() > 0)
    {
      if (get_constant_value(
            len_node["referencedDeclaration"].get<int>(), length))
        length.clear();
    }

    if (length.empty())
    {
      // Could not resolve the fixed length — degrade to dynamic array
      // so the frontend does not crash on huge/unresolvable sizes.
      set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
      return false;
    }
    // Solidity allows arbitrarily large static array sizes (e.g.
    // `uint[2**253]`), which blow up downstream when GOTO/SMT tries to
    // materialize the element-count-by-element-width product. Clang's
    // C frontend caps static arrays around 2^58 elements; match that
    // and degrade to DYNARRAY above the cap so verification can still
    // proceed on benign code paths that never actually index such an
    // array with a concrete large value.
    {
      constexpr unsigned long long kMaxStaticArrayElems = 1ULL << 56;
      bool too_big = length.size() > 19; // > max uint64 digit count
      if (!too_big)
      {
        try
        {
          unsigned long long n = std::stoull(length);
          if (n > kMaxStaticArrayElems)
            too_big = true;
        }
        catch (const std::exception &)
        {
          too_big = true;
        }
      }
      if (too_big)
      {
        log_warning(
          "Solidity array length {} exceeds supported static size; "
          "modelling as dynamic array",
          length);
        set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
        return false;
      }
    }
    new_type.set("#sol_array_size", length);
    set_sol_type(new_type, SolidityGrammar::SolType::ARRAY);
  }
  else
    set_sol_type(new_type, SolidityGrammar::SolType::DYNARRAY);
  return false;
}

bool solidity_convertert::is_byte_type(const typet &t)
{
  if (SolidityGrammar::is_bytes_type(get_sol_type(t)))
    return true;
  if (
    t.is_struct() &&
    (t.type().tag() == "BytesDynamic" || t.type().tag() == "BytesStatic"))
    return true;
  return false;
}

bool solidity_convertert::is_bytesN_type(const typet &t)
{
  SolidityGrammar::SolType solt = get_sol_type(t);
  if (solt == SolidityGrammar::SolType::BYTES_STATIC)
    return true;
  if (t.is_struct() && t.type().tag() == "BytesStatic")
    return true;
  return false;
}

bool solidity_convertert::is_bytes_type(const typet &t)
{
  // expects t like "bytes" (dynamic)
  SolidityGrammar::SolType solt = get_sol_type(t);
  if (solt == SolidityGrammar::SolType::BYTES_DYN)
    return true;
  if (t.is_struct() && t.type().tag() == "BytesDynamic")
    return true;
  return false;
}

void solidity_convertert::convert_type_expr(
  const namespacet &ns,
  exprt &src_expr,
  const typet &dest_type,
  const nlohmann::json &expr)
{
  exprt _null = nil_exprt();
  _null.type() = dest_type;
  convert_type_expr(ns, src_expr, _null, expr);
}

void solidity_convertert::convert_type_expr(
  const namespacet &ns,
  exprt &src_expr,
  const exprt &dest_expr,
  const nlohmann::json &expr)
{
  log_debug("solidity", "\t@@@ Performing type conversion");

  typet src_type = src_expr.type();
  typet dest_type = dest_expr.type();
  SolidityGrammar::SolType src_sol_type = get_sol_type(src_type);
  SolidityGrammar::SolType dest_sol_type = get_sol_type(dest_type);

  bool not_same_type = false;
  if (src_type != dest_type)
    not_same_type = true;
  else if (
    src_sol_type != SolidityGrammar::SolType::UNSET &&
    dest_sol_type != SolidityGrammar::SolType::UNSET)
  {
    if (src_sol_type != dest_sol_type)
      not_same_type = true;
    else if (
      src_type.get("#sol_bytesn_size") != dest_type.get("#sol_bytesn_size"))
      // including unset situation
      not_same_type = true;
    else if (
      src_type.get("#sol_array_size") != dest_type.get("#sol_array_size"))
      not_same_type = true;
  }

  // only do conversion when the src.type != dest.type
  if (not_same_type)
  {
    log_debug(
      "solidity",
      "\t\tGot src_sol_type = {}",
      SolidityGrammar::sol_type_to_str(src_sol_type));
    if (src_sol_type == SolidityGrammar::SolType::UNSET)
      log_debug("solidity", "{}", src_type.to_string());
    log_debug(
      "solidity",
      "\t\tGot dest_sol_type = {}",
      SolidityGrammar::sol_type_to_str(dest_sol_type));
    if (dest_sol_type == SolidityGrammar::SolType::UNSET)
      log_debug("solidity", "{}", dest_type.to_string());

    if (is_byte_type(src_type) && is_byte_type(dest_type))
    {
      // prevent something like
      // bytes_dynamic_from_uint({ .offset=0, .length=0, .initialized=0, .anon_pad$3=0 }, this->$dynamic_pool);
      if (src_expr.is_struct())
        src_expr = make_aux_var(src_expr, src_expr.location());

      exprt pool_member;
      if (get_dynamic_pool(expr, pool_member))
        abort();

      // e.g. Bytes2 x; Bytes4(x); -> bytes_static_truncate(&x, 2)
      // Bytes2 y; Bytes4 x = Bytes4(y);
      if (is_bytesN_type(src_type) && is_bytesN_type(dest_type))
      {
        side_effect_expr_function_callt resize_call;
        get_library_function_call_no_args(
          "bytes_static_resize",
          "c:@F@bytes_static_resize",
          dest_type,
          src_expr.location(),
          resize_call);

        exprt len_expr;
        get_bytesN_size(dest_expr, len_expr);
        resize_call.arguments().push_back(src_expr);
        resize_call.arguments().push_back(len_expr);

        src_expr = make_aux_var(resize_call, src_expr.location());
        set_sol_type(src_expr.type(), SolidityGrammar::SolType::BYTES_STATIC);
        return;
      }

      // e.g. bytes2 x; bytes y = bytes(x);
      else if (is_bytesN_type(src_type) && is_bytes_type(dest_type))
      {
        side_effect_expr_function_callt from_static_call;
        assert(
          context.find_symbol("c:@F@bytes_dynamic_from_static") != nullptr);
        get_library_function_call_no_args(
          "bytes_dynamic_from_static",
          "c:@F@bytes_dynamic_from_static",
          dest_type,
          src_expr.location(),
          from_static_call);
        from_static_call.arguments().push_back(src_expr);
        from_static_call.arguments().push_back(pool_member);

        src_expr = make_aux_var(from_static_call, src_expr.location());
        set_sol_type(src_expr.type(), SolidityGrammar::SolType::BYTES_DYN);
        return;
      }

      // e.g. bytes x; bytes2 y = bytes2(x);
      else if (is_bytes_type(src_type) && is_bytesN_type(dest_type))
      {
        side_effect_expr_function_callt resize_dyn_call;
        get_library_function_call_no_args(
          "bytes_static_resize_from_dynamic",
          "c:@F@bytes_static_resize_from_dynamic",
          dest_type,
          src_expr.location(),
          resize_dyn_call);

        exprt len_expr;
        get_bytesN_size(dest_expr, len_expr);
        resize_dyn_call.arguments().push_back(src_expr);
        resize_dyn_call.arguments().push_back(len_expr);
        resize_dyn_call.arguments().push_back(pool_member);

        src_expr = make_aux_var(resize_dyn_call, src_expr.location());
        set_sol_type(src_expr.type(), SolidityGrammar::SolType::BYTES_STATIC);
        return;
      }

      // e.g. bytes x; bytes y = bytes(x);
      else
      {
        side_effect_expr_function_callt copy_call;
        assert(context.find_symbol("c:@F@bytes_dynamic_copy") != nullptr);
        get_library_function_call_no_args(
          "bytes_dynamic_copy",
          "c:@F@bytes_dynamic_copy",
          dest_type,
          src_expr.location(),
          copy_call);
        copy_call.arguments().push_back(src_expr);
        copy_call.arguments().push_back(pool_member);

        src_expr = make_aux_var(copy_call, src_expr.location());
        set_sol_type(src_expr.type(), SolidityGrammar::SolType::BYTES_DYN);
        return;
      }
    }
    // int/symbol to bytes or bytesN
    else if (!is_byte_type(src_type) && is_byte_type(dest_type))
    {
      // this could be
      // bytes(hex"1234") -> string literal
      // bytes("1234") -> string literal
      // byte4("123") -> string
      // bytes(x)  -> string literal
      // bytes2(0x1234) -> int literal
      // uint256-result -> bytes (e.g. abi.encode/encodeCall lowered to
      //                  the uint256 identity in solidity_abi.c being
      //                  used in a `bytes` context)

      locationt loc = src_expr.location();
      if (is_bytes_type(dest_type))
      {
        // bytes_dynamic_from_string expects a `const char *`. A scalar
        // (e.g. the abi.encode* identity uint256, or any other non-pointer
        // value being narrowed to bytes) cannot be safely dereferenced
        // through that path — it would either crash symex with an invalid
        // dereference or silently read garbage. Fall back to the bounded
        // nondet bytes harness used by other approximation sites — sound
        // OVER-approximation: "the encoder produced some bytes of length
        // ∈ [32, 1024] with initialized==1". Same rationale as the
        // string-constant fallback in the return-statement handler.
        if (!src_type.is_pointer() && !src_type.is_array())
        {
          side_effect_expr_function_callt nondet_b;
          get_library_function_call_no_args(
            "llc_nondet_bytes",
            "c:@F@llc_nondet_bytes",
            dest_type,
            loc,
            nondet_b);
          src_expr = make_aux_var(nondet_b, loc);
          set_sol_type(
            src_expr.type(), SolidityGrammar::SolType::BYTES_DYN);
          return;
        }

        side_effect_expr_function_callt call;
        get_library_function_call_no_args(
          "bytes_dynamic_from_string",
          "c:@F@bytes_dynamic_from_string",
          dest_type,
          loc,
          call);
        src_expr = make_aux_var(src_expr, src_expr.location());
        call.arguments().push_back(src_expr);
        set_sol_type(call.type(), SolidityGrammar::SolType::BYTES_DYN);

        // resolve pool_data: this.dynamic_pool
        exprt pool_member;
        if (get_dynamic_pool(expr, pool_member))
          abort();
        call.arguments().push_back(pool_member);

        src_expr = make_aux_var(call, loc);
      }
      else if (is_bytesN_type(dest_type))
      {
        side_effect_expr_function_callt call;
        get_library_function_call_no_args(
          "bytes_static_from_uint",
          "c:@F@bytes_static_from_uint",
          dest_type,
          loc,
          call);
        call.arguments().push_back(src_expr);

        // e.g. bytes3(0x1234); "BYTES3" => 3
        exprt len_expr;
        if (!dest_type.get("#sol_bytesn_size").empty())
          len_expr = from_integer(
            std::stoul(dest_type.get("#sol_bytesn_size").as_string()),
            size_type());
        else
        {
          log_error("got unexpected bytes typecast");
          abort();
        }
        call.arguments().push_back(len_expr);

        src_expr = make_aux_var(call, loc);
        set_sol_type(src_expr.type(), SolidityGrammar::SolType::BYTES_STATIC);
      }
      else
      {
        log_error(
          "Unknown bytes destination type: {}",
          SolidityGrammar::sol_type_to_str(dest_sol_type));
        abort();
      }
    }
    else if (is_byte_type(src_type) && dest_type.is_unsignedbv())
    {
      side_effect_expr_function_callt call;
      locationt loc = src_expr.location();

      if (is_bytesN_type(src_type))
      {
        get_library_function_call_no_args(
          "bytes_static_to_uint",
          "c:@F@bytes_static_to_uint",
          dest_type,
          loc,
          call);
        call.arguments().push_back(src_expr);
      }
      else if (is_bytes_type(src_type))
      {
        get_library_function_call_no_args(
          "bytes_dynamic_to_uint",
          "c:@F@bytes_dynamic_to_uint",
          dest_type,
          loc,
          call);
        call.arguments().push_back(src_expr);

        exprt pool_member;
        if (get_dynamic_pool(expr, pool_member))
          abort();
        call.arguments().push_back(pool_member);
      }
      else
      {
        log_error("Expected bytes or bytesN for to_uint conversion");
        abort();
      }

      src_expr = call;
      return;
    }
    // string(bytes)
    else if (
      is_byte_type(src_type) && dest_type.is_pointer() &&
      dest_type.subtype().is_signedbv())
    {
      locationt loc = src_expr.location();
      exprt call;

      if (is_bytesN_type(src_type))
      {
        side_effect_expr_function_callt fn_call;
        get_library_function_call_no_args(
          "bytes_static_to_string",
          "c:@F@bytes_static_to_string",
          dest_type,
          loc,
          fn_call);
        fn_call.arguments().push_back(src_expr);

        call = fn_call;
      }
      else if (is_bytes_type(src_type))
      {
        side_effect_expr_function_callt fn_call;
        get_library_function_call_no_args(
          "bytes_dynamic_to_string",
          "c:@F@bytes_dynamic_to_string",
          dest_type,
          loc,
          fn_call);
        fn_call.arguments().push_back(src_expr);

        exprt pool_member;
        if (get_dynamic_pool(expr, pool_member))
          abort();
        fn_call.arguments().push_back(pool_member);

        call = fn_call;
      }
      else
      {
        log_error("Expected bytes or bytesN for to_string conversion");
        abort();
      }

      src_expr = call;
      return;
    }
    else if (
      (SolidityGrammar::is_address_type(dest_sol_type)) &&
      (src_sol_type == SolidityGrammar::SolType::CONTRACT || src_sol_type == SolidityGrammar::SolType::UNSET))
    {
      // CONTRACT: address(instance) ==> instance.address
      // EMPTY: address(this) ==> this.address
      std::string comp_name = "$address";
      typet t;
      if (dest_sol_type == SolidityGrammar::SolType::ADDRESS)
        t = addr_t;
      else
        t = addrp_t;

      // Only build the synthetic $address member when src_expr's
      // resolved struct actually carries it (a contract instance).
      // Inside a `library` function body src_expr is the library
      // struct, which has no $address. A library executes via
      // DELEGATECALL in the caller's context, so address(this) is the
      // enclosing contract's address — model it with the ambient
      // _ESBMC_enclosing_contract_address (same choice as the library
      // branch in solidity_convert_call.cpp ~3336). Without this guard
      // member_exprt(<library struct>, "$address") survives the
      // frontend and aborts at migrate.cpp -> irep2_type.cpp
      // get_component_number during goto_convert.
      if (struct_type_has_component(src_expr.type(), comp_name))
        src_expr = member_exprt(src_expr, comp_name, t);
      else
      {
        exprt encl = symbol_expr(
          *context.find_symbol("c:@_ESBMC_enclosing_contract_address"));
        solidity_gen_typecast(ns, encl, t);
        src_expr = encl;
      }
    }
    else if (
      (SolidityGrammar::is_address_type(src_sol_type) ||
       (src_sol_type == SolidityGrammar::SolType::UNSET &&
        src_type.is_unsignedbv())) &&
      dest_sol_type == SolidityGrammar::SolType::CONTRACT)
    {
      // E.g. for `Derive x = Derive(_addr)`:
      // => Derive* x = &_ESBMC_Obeject_Derive;
      // because in trusted mode, the address has been limited to the set of _ESBMC_Object
      // Save the original address before overwriting src_expr
      exprt original_addr = src_expr;

      exprt c_ins;
      std::string _cname = dest_type.get("#sol_contract").as_string();
      // A library is converted in round 1, before any contract/interface
      // struct type exists (round 2). `IFoo(addr)` inside a library body then
      // referenced tag-IFoo before it was registered, and the implicit
      // typecast that follows ns.follow()ed the missing symbol: SIGSEGV in
      // irept::read (perennial Product: PayoffDefinitionLib._providerContract
      // `IContractPayoffProvider(address(bytes20(...)))`; minimal: a library
      // `function f(address a) internal pure returns (I) { return I(a); }`).
      // Convert that contract on demand; get_contract_definition saves and
      // restores the current conversion context for exactly this re-entry.
      // Bases first (most-base to most-derived): converting a derived
      // interface registers builtins on its bases (move_builtin_to_contract
      // aborts with "parsing order error" when a base is not registered yet).
      if (!_cname.empty() && context.find_symbol(prefix + _cname) == nullptr)
      {
        std::vector<std::string> order;
        std::map<int, std::string> id2name;
        const nlohmann::json *def = nullptr;
        for (const auto &n : src_ast_json["nodes"])
          if (n.value("nodeType", "") == "ContractDefinition")
          {
            id2name[n["id"].get<int>()] = n["name"].get<std::string>();
            if (n["name"] == _cname)
              def = &n;
          }
        if (def && def->contains("linearizedBaseContracts"))
          for (auto it = (*def)["linearizedBaseContracts"].rbegin();
               it != (*def)["linearizedBaseContracts"].rend();
               ++it)
            if (id2name.count(it->get<int>()))
              order.push_back(id2name[it->get<int>()]);
        if (order.empty())
          order.push_back(_cname);
        for (const auto &b : order)
          if (
            context.find_symbol(prefix + b) == nullptr &&
            get_contract_definition(b))
          {
            log_error(
              "failed to convert contract {} referenced by a cast", b);
            abort();
          }
      }
      get_static_contract_instance_ref(_cname, c_ins);

      // Propagate the cast address into the singleton's $address member
      // so that address(ContractType(addr)) == addr holds.
      member_exprt addr_member(c_ins, "$address", addr_t);
      solidity_gen_typecast(ns, original_addr, addr_t);
      exprt assign_addr = side_effect_exprt("assign", addr_t);
      assign_addr.copy_to_operands(addr_member, original_addr);
      convert_expression_to_code(assign_addr);
      move_to_front_block(assign_addr);

      // type conversion
      src_expr = address_of_exprt(c_ins);
      set_sol_type(src_expr.type(), SolidityGrammar::SolType::CONTRACT);
    }
    else if (
      (src_sol_type == SolidityGrammar::SolType::ARRAY_LITERAL) && src_type.id() == typet::id_array)
    {
      // this means we are handling a constant array
      // which should be assigned to an array pointer
      // e.g. data1 = [int8(6), 7, -8, 9, 10, -12, 12];

      log_debug("solidity", "\t@@@ Converting array literal to symbol");

      if (dest_type.id() != typet::id_pointer)
      {
        log_error(
          "Expecting dest_type to be pointer type, got = {}",
          dest_type.id().as_string());
        abort();
      }

      // dynamic: uint x[] = [1,2]
      // fixed:   uint x[3] = [1,2], whose rhs array is incomplete and need to add zero element
      // the goal is to convert the rhs constant array to a static global var

      // get rhs constant array size
      const std::string src_size = src_type.get("#sol_array_size").as_string();
      if (src_size.empty())
      {
        // e.g. a = new uint[](len);
        // we have already populate the auxiliary state var so
        // skip the rest of the process
        // ? solidity_gen_typecast(ns, src_expr, dest_type);
        return;
      }
      unsigned z_src_size = std::stoul(src_size, nullptr);

      // get lhs array size
      std::string dest_size = dest_type.get("#sol_array_size").as_string();
      if (dest_size.empty())
      {
        if (dest_sol_type == SolidityGrammar::SolType::ARRAY)
        {
          log_error("Unexpected empty-length fixed array");
          abort();
        }
        // the dynamic array does not have a fixed length
        // therefore set it as the rhs length
        dest_size = src_size;
      }
      unsigned z_dest_size = std::stoul(dest_size, nullptr);
      constant_exprt dest_array_size = constant_exprt(
        integer2binary(z_dest_size, bv_width(int_type())),
        integer2string(z_dest_size),
        int_type());

      if (src_expr.id() == irept::id_member)
      {
        // e.g. uint[3] x;  (x, y) = ([1,z], ...)
        // where [1,2] ==> uint8[] ==> tuple_instance.mem0
        // ==>
        //  x  = [(uint256)tuple_instance.mem0[0], (uint256)tuple_instance.mem0[1], 0]
        // - src_expr: [1, z]
        // - dest_type: uint*
        array_typet arr_t = array_typet(dest_type.subtype(), dest_array_size);
        set_sol_type(arr_t, SolidityGrammar::SolType::ARRAY);
        arr_t.set("#sol_array_size", src_size);
        exprt new_arr = exprt(irept::id_array, arr_t);

        exprt arr_comp;
        for (unsigned i = 0; i < z_src_size; i++)
        {
          // do array index
          exprt idx = constant_exprt(
            integer2binary(i, bv_width(size_type())),
            integer2string(i),
            size_type());
          exprt op = index_exprt(src_expr, idx, src_type.subtype());

          arr_comp = typecast_exprt(op, dest_type.subtype());
          new_arr.operands().push_back(arr_comp);
        }

        src_expr = new_arr;
      }
      else if (src_expr.id() != irept::id_array)
      {
        // Runtime fixed-array reference (id_index for `s[1]`, id_symbol for
        // `local_arr`, etc.). The aux-array materialization below assumes
        // src_expr is a literal whose operands can become a static
        // initializer; for live expressions that path produces a
        // static-lifetime symbol whose `value` references function-local
        // state (e.g. `((uint256)(&s[0][0]))[1]`) and migrate_expr crashes
        // on the type-mismatched index2t source. Decay to a pointer the
        // normal C way and skip the rest of this branch.
        solidity_gen_typecast(ns, src_expr, dest_type);
        return;
      }

      // allow fall-through
      if (src_expr.id() == irept::id_array)
      {
        log_debug("solidity", "\t@@@ Populating zero elements to array");

        // e.g. uint[3] x = [1] ==> uint[3] x == [1,0,0]
        unsigned s_size = src_expr.operands().size();
        if (s_size != z_src_size)
        {
          log_error(
            "Expecting equivalent array size, got {} and {}",
            std::to_string(s_size),
            std::to_string(z_src_size));
          abort();
        }
        if (z_dest_size > s_size)
        {
          for (unsigned i = 0; i < s_size; i++)
          {
            exprt &op = src_expr.operands().at(i);
            solidity_gen_typecast(ns, op, dest_type.subtype());
          }
          exprt _zero =
            gen_zero(get_complete_type(dest_type.subtype(), ns), true);
          _zero.location() = src_expr.location();
          _zero.set("#cformat", 0);
          // push zero
          for (unsigned i = s_size; i < z_dest_size; i++)
            src_expr.operands().push_back(_zero);

          // reset size
          assert(src_expr.type().is_array());
          to_array_type(src_expr.type()).size() = dest_array_size;

          // update "#sol_array_size"
          assert(!dest_size.empty());
          src_expr.type().set("#sol_array_size", dest_size);
        }
      }

      // since it's a array-constant/string-constant, we could safely make it to a local var
      // this local var will not be referred again so the name could be random.
      // e.g.
      // int[3] p = [1,2];
      // => int *p = [1,2,3];
      // => static int[3] tmp1 = [1,2,3];
      // return: src_expr = symbol_expr(tmp1)
      exprt new_expr;
      get_aux_array(src_expr, dest_type.subtype(), new_expr);
      src_expr = new_expr;
    }
    else
      solidity_gen_typecast(ns, src_expr, dest_type);
  }
}

