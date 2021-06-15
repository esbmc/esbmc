#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_

#include <nlohmann/json.hpp>
#include <solidity-ast-frontend/solidity_type.h>
#include <iostream>
#include <iomanip>
#include <vector>

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
  bool get_isConstQualified() const { return isConstQualified; }
  bool get_isVolatileQualified() const { return isVolatileQualified; }
  bool get_isRestrictQualified() const { return isRestrictQualified; }

private:
  SolidityTypes::typeClass type_class;
  SolidityTypes::builInTypesKind bt_kind; // builtinType::getKind();
  bool isConstQualified;
  bool isVolatileQualified;
  bool isRestrictQualified;
};

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
  }

  // for debug print
  void print_decl_json();

  // config this tracker based on json values
  void config();

  // getters
  SolidityTypes::declKind get_decl_kind()   { return decl_kind; }
  QualTypeTracker& get_qualtype_tracker()   { return qualtype_tracker; }
  NamedDeclTracker& get_nameddecl_tracker() { return nameddecl_tracker; }
  bool get_hasAttrs()                       { return hasAttrs; }

private:
  nlohmann::json decl_json;
  SolidityTypes::declKind decl_kind; // decl.getKind()
  QualTypeTracker qualtype_tracker;
  NamedDeclTracker nameddecl_tracker;
  bool hasAttrs;

  // private setters : set the member values based on the corresponding json value. Used by config() only.
  // Setting them outside this class is NOT allowed.
  void set_decl_kind();
  // QualTypeTracker setters
  void set_qualtype_tracker();
  void set_qualtype_bt_kind();
  // NamedDeclTracker setters
  void set_named_decl_name();
  void set_named_decl_kind();
};

#endif // END of SOLIDITY_AST_FRONTEND_SOLIDITY_DECL_TRACKER_H_
