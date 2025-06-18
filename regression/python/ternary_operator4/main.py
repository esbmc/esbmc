x = 10 
y = 5
result:int = x + y if x > y else x * y

is_raining = True
activity:str = "Stay inside and read" if is_raining else "Go for a walk"
print(f"Raining: {is_raining} → {activity}")

temperature = 25
sunny = True
weather_advice:str = "Perfect beach weather!" if temperature > 20 and sunny else "Maybe stay inside"
print(f"Temp: {temperature}°C, Sunny: {sunny} → {weather_advice}")
