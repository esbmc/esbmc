#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_

#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>
#include <iostream>
#include <iomanip>
#include <vector>

// declaration's source manager for location settings
class SourceLocationTracker
{
public:
  static constexpr unsigned lineNumberInvalid = std::numeric_limits<unsigned>::max();

  SourceLocationTracker() { clear_all(); }

  void clear_all()
  {
    line_number = lineNumberInvalid;
    file_name = "";
    isFunctionOrMethod = false;
    isValid = false;
  }

  // setter
  void set_line_number(unsigned _ln)   { line_number = _ln; }
  void set_file_name(std::string _fn)  { file_name = _fn; }
  void set_isFunctionOrMethod(bool _v) { isFunctionOrMethod = _v; };
  void set_isValid(bool _v)            { isValid = _v; }

  // getter
  unsigned get_line_number()  const   { return line_number; }
  std::string get_file_name() const   { return file_name; }
  bool get_isFunctionOrMethod() const { return isFunctionOrMethod; };
  bool get_isValid() const            { return isValid; }

private:
  unsigned line_number;
  std::string file_name;
  bool isFunctionOrMethod; // is upstream decl a function
  bool isValid; // if true, it means a valid sm pointer as in "sm = &ASTContext->getSourceManager();"
};

// declaration's identifier manager for name and id settings
class NamedDeclTracker
{
public:
  NamedDeclTracker() { clear_all(); }

  void clear_all()
  {
    name = "";
    named_decl_kind = SolidityTypes::DeclKindError;
    hasIdentifier = false;
  }

  // setter
  void set_name(std::string _name) { name = _name; }
  void set_named_decl_kind(SolidityTypes::declKind _kind) { named_decl_kind = _kind; }
  void set_hasIdentifier(bool _has) { hasIdentifier = _has; }

  // getter
  SolidityTypes::declKind get_named_decl_kind() const { return named_decl_kind; }
  std::string get_name() const   { return name; }
  bool get_hasIdentifier() const { return hasIdentifier; }

private:
  std::string name;
  SolidityTypes::declKind named_decl_kind; // same as Xdecl.getKind(), where X can be a VarDecl or other types of Decl
  bool hasIdentifier; // indicate this declaration has a name, == const clang::IdentifierInfo *identifier = nd.getIdentifier()
};

// declaration's subtype (e.g. builtin_kind) manager for subtype settings
class QualTypeTracker
{
public:
  QualTypeTracker() { clear_all(); }

  void clear_all()
  {
    type_class = SolidityTypes::TypeError;
    bt_kind = SolidityTypes::BuiltInError;
    isConstQualified = false; // TODO: what's Solidity's equivalent? Hard coded for now.
    isVolatileQualified = false;
    isRestrictQualified = false;
  }

  // setter
  void set_type_class(SolidityTypes::typeClass _type)    { type_class = _type; }
  void set_bt_kind(SolidityTypes::builInTypesKind _kind) { bt_kind    = _kind; }
  void set_isConstQualified(bool _v)    { isConstQualified = _v; }
  void set_isVolatileQualified(bool _v) { isVolatileQualified = _v; }
  void set_isRestrictQualified(bool _v) { isRestrictQualified = _v; }

  // getter
  SolidityTypes::typeClass get_type_class() const    { return type_class; }
  SolidityTypes::builInTypesKind get_bt_kind() const { return bt_kind; };
  bool get_isConstQualified() const    { return isConstQualified; }
  bool get_isVolatileQualified() const { return isVolatileQualified; }
  bool get_isRestrictQualified() const { return isRestrictQualified; }

private:
  SolidityTypes::typeClass type_class;
  SolidityTypes::builInTypesKind bt_kind; // builtinType::getKind();
  bool isConstQualified;
  bool isVolatileQualified;
  bool isRestrictQualified;
};

// variable declaration
class VarDeclTracker
{
public:
  VarDeclTracker(nlohmann::json _decl_json):
    decl_json(_decl_json)
  {
      clear_all();
  }

  VarDeclTracker(const VarDeclTracker &rhs) = default; // default copy constructor
  // move constructor
  VarDeclTracker(VarDeclTracker &&rhs) = default;
  // copy assignment operator - TODO: Since we have a reference member, it can be tricky in this case.
  //    Let's not do it for the time being. Leave it for future improvement
  VarDeclTracker& operator=(const VarDeclTracker &rhs) { assert(!"copy assignment is not allowed at the moment"); }
  // move assignment operator - TODO: Since we have a reference member, it can be tricky in this case.
  //    Let's not do it for the time being. Leave it for future improvement
  VarDeclTracker& operator=(VarDeclTracker &&rhs) { assert(!"move assignment is not allowed at the moment"); }

  void clear_all() // reset to default values
  {
    decl_kind = SolidityTypes::DeclKindError;
    hasAttrs = false;
    absolute_path = "";
    storage_class = SolidityTypes::SCError;
    hasGlobalStorage = false;
    hasExternalStorage = false;
    isExternallyVisible = false;
    hasInit = false;
  }

  // for debug print
  void print_decl_json();

  // config this tracker based on json values
  void config(std::string ab_path);

  // getters
  SolidityTypes::declKind get_decl_kind()   { return decl_kind; }
  QualTypeTracker& get_qualtype_tracker()   { return qualtype_tracker; }  // for get_type(...)
  NamedDeclTracker& get_nameddecl_tracker() { return nameddecl_tracker; } // for get_decl_name(...)
  SourceLocationTracker& get_sl_tracker()   { return sl_tracker; }        // for get_decl_name(...)
  bool get_hasAttrs()                       { return hasAttrs; }
  bool get_hasGlobalStorage()               { return hasGlobalStorage; }
  bool get_hasExternalStorage()             { return hasExternalStorage; }
  bool get_isExternallyVisible()            { return isExternallyVisible; }
  bool get_hasInit()                        { return hasInit; }
  std::string get_absolute_path()           { return absolute_path; }
  SolidityTypes::storageClass get_storage_class() { return storage_class; }

private:
  nlohmann::json decl_json;
  SolidityTypes::declKind decl_kind; // decl.getKind()
  QualTypeTracker qualtype_tracker;
  NamedDeclTracker nameddecl_tracker;
  SourceLocationTracker sl_tracker;
  bool hasAttrs;
  bool hasGlobalStorage;
  bool hasExternalStorage;
  bool isExternallyVisible; // NOT a direct translation of Solidity's visibility. Visible to other functions in this contract
  bool hasInit;
  std::string absolute_path;
  SolidityTypes::storageClass storage_class;

  // private setters : set the member values based on the corresponding json value. Used by config() only.
  // Setting them outside this class is NOT allowed.
  void set_decl_kind();
  // QualTypeTracker setters
  void set_qualtype_tracker();
  void set_qualtype_bt_kind();
  // NamedDeclTracker setters
  void set_named_decl_name();
  void set_named_decl_kind();
  // SourceLocationTracker setters
  void set_sl_tracker_line_number();
  void set_sl_tracker_file_name();
  // static lifetime, extern, file_local setters
  void set_hasGlobalStorage();
  // init value setters
  void set_hasInit();
};

#endif // END of SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
