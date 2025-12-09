enum class Color;

void setColor(Color color) {
}

enum class Color {
    RED,
    GREEN,
    BLUE
};

int main() {
    setColor(Color::BLUE);
    return 0;
}
