#pragma once

#include <python-frontend/symbol_id.h>
#include <util/symbol.h>
#include <util/context.h>
#include <nlohmann/json.hpp>
#include <regex>

class symbol_finder
{
public:
  symbol_finder(const contextt &symbol_table, const nlohmann::json ast)
    : symbol_table_(symbol_table), ast_(ast)
  {
  }

  const symbolt *find_symbol(const std::string &symbol_id)
  {
    if (const symbolt *symbol = symbol_table_.find_symbol(symbol_id))
      return symbol;

    if (const symbolt *symbol = find_symbol_in_global_scope(symbol_id))
      return symbol;

    return find_imported_symbol(symbol_id);
  }

private:
  const symbolt *find_symbol_in_global_scope(const std::string &symbol_id)
  {
    std::size_t class_start_pos = symbol_id.find("@C@");
    std::size_t func_start_pos = symbol_id.find("@F@");
    std::string sid = symbol_id;

    // Remove class name from symbol
    if (class_start_pos != std::string::npos)
      sid.erase(class_start_pos, func_start_pos - class_start_pos);

    func_start_pos = sid.find("@F@");
    std::size_t func_end_pos = sid.rfind("@");

    // Remove function name from symbol
    if (func_start_pos != std::string::npos)
      sid.erase(func_start_pos, func_end_pos - func_start_pos);

    return symbol_table_.find_symbol(sid);
  }

  const symbolt *find_imported_symbol(const std::string &symbol_id)
  {
    for (const auto &obj : ast_["body"])
    {
      if (obj["_type"] == "ImportFrom" || obj["_type"] == "Import")
      {
        std::regex pattern("py:(.*?)@");
        std::string imported_symbol = std::regex_replace(
          symbol_id,
          pattern,
          "py:" + obj["full_path"].get<std::string>() + "@");

        if (
          const symbolt *func_symbol =
            symbol_table_.find_symbol(imported_symbol.c_str()))
          return func_symbol;
      }
    }
    return nullptr;
  }

  const contextt &symbol_table_;
  const nlohmann::json &ast_;
};
