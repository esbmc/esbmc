/* T5: try/catch with user-defined exceptions, exercising exact-type and
   supertype handler matching. Avoids division by zero and array indexing:
   JBMC checks both as properties in their own right, so a deliberately
   thrown ArithmeticException fails verification even when it is caught. */
public class T5Exception {
  static class Base extends Exception {
    private static final long serialVersionUID = 1L;
  }

  static class Derived extends Base {
    private static final long serialVersionUID = 1L;
  }

  static void raise(boolean derived) throws Base {
    if (derived) throw new Derived();
    throw new Base();
  }

  public static void main(String[] args) {
    int exact = 0;
    try {
      raise(true);
    } catch (Derived e) {
      exact = 1;
    } catch (Base e) {
      exact = 2;
    }
    assert exact == 1;

    int viaSuper = 0;
    try {
      raise(false);
    } catch (Base e) {
      viaSuper = 1;
    }
    assert viaSuper == 1;
  }
}
