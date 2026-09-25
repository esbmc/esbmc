/* T4: inheritance, virtual dispatch and instanceof. */
package jbmcpoc;

public class T4Virtual {
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
    assert s.area() == 25;
    assert s instanceof Square;
    Shape plain = new Shape();
    assert plain.area() == 0;
  }
}
