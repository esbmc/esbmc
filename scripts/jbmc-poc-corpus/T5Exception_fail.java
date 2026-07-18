/* T5 negative: the handler runs, but the asserted outcome is wrong. */
public class T5Exception_fail {
  static class Base extends Exception {
    private static final long serialVersionUID = 1L;
  }

  static void raise() throws Base {
    throw new Base();
  }

  public static void main(String[] args) {
    int caught = 0;
    try {
      raise();
    } catch (Base e) {
      caught = 1;
    }
    assert caught == 2;
  }
}
