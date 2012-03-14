
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

  /** Instance of expr_ids recording tihs exprs type. */
  expr_ids expr_id;

  /** Type of this expr. All exprs have a type. */
  const type2t &type;
};
