/* T4 negative: the override returns a different value than asserted. */
public class T4Virtual_fail {
  static class Shape {
    int area() { return 0; }
  }

  static class Square extends Shape {
    int side;

    Square(int s) { this.side = s; }

    @Override int area() { return side * side; }
  }

  public static void main(String[] args) {
    Shape s = new Square(5);
    assert s.area() == 24;
  }
}
