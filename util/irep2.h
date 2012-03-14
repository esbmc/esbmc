#include <vector>

#include <big-int/bigint.hh>

/** Base class for all types */
class type2t
{
protected:
  type2t(type_ids id);
  type2t(const type2t &ref);

public:
  /** Enumeration identifying each sort of type.
   *  The idea being that we might (for whatever reason) at runtime need to fall
   *  back onto identifying a type through just one field, for some reason. It's
   *  also highly useful for debugging */
  enum type_ids {
  };

  /** Instance of type_ids recording this types type. */
  type_ids type_id;
};

/** Base class for all expressions */
class expr2t
{
protected:
  expr2t(type2t &type, expr_ids id);
  expr2t(const expr2t &ref);

public:
  /** Enumeration identifying each sort of expr.
   *  The idea being to permit runtime identification of a type, for debugging or
   *  otherwise. See type2t::type_ids. */
  enum expr_ids {
  };

  /** Clone method. Entirely self explanatory */
  virtual expr2t *clone(void) const = 0;

  /** Instance of expr_ids recording tihs exprs type. */
  expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  const type2t &type;
};

/** Constant class type. Not designed to contain any piece of data or method in
 *  particular, but allows for additional type safety. */
class constant2t : expr2t
{
protected:
  constant2t(type2t &type, expr_ids id);
  constant2t(const constant2t &ref);

public:
  /** Clone method. Entirely self explanatory */
  virtual expr2t *clone(void) const = 0;
};

/** Constant integer class. Records a constant integer of an arbitary
 *  precision */
class constant_int2t : constant2t
{
protected:
  constant_int2t(const BigInt &input);
  constant_int2t(const constant_int2t &ref);

  /** Concrete clone implementation. */
  virtual expr2t *clone(void) const;

  /** Arbitary precision integer record. */
  BigInt constant_value;
};

/** Constant class for string constants. */
class constant_string2t : constant2t
{
protected:
  constant_string2t(const std::string stringref &ref);
  constant_string2t(const constant_string2t &ref);

public:
  /** Concrete clone implementation. */
  virtual expr2t *clone(void) const;

  /** Arbitary precision integer record. */
  const std::string value;
};

/** Const datatype - for holding structs and unions */
class constant_datatype2t : constant2t
{
protected:
  constant_datatype2t(const type2t &type, const std::vector<exprt *> &members);
  constant_datatype2t(const constant_datatype2t &ref);

public:
  virtual expr2t *clone(void) const = 0;

  std::vector<exprt *> datatype_members;
};

class constant_struct2t : constant_datatype2t
{
protected:
  constant_struct2t(const type2t &type, const std::vector<exprt *> &members);
  constant_struct2t(const constant_struct2t &ref);

public:
  virtual expr2t *clone(void) const;
};

class constant_union2t : constant_datatype2t
{
protected:
  constant_union2t(const type2t &type, const std::vector<exprt *> &members);
  constant_union2t(const constant_union2t &ref);

public:
  virtual expr2t *clone(void) const;
};
