fun tileColors() : Array<Array<Int>> {
  val shades = Array(2) {
	Array(2) { 0 }
  }

  shades[0][0] = 192
  shades[0][1] = 128
  shades[1][0] = 64
  shades[1][1] = 0
  shades[3][2] = 54 //out-of-bounds
  return shades
}

fun main() {
  tileColors()
}
