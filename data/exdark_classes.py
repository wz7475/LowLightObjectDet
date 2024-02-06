class_string = "Bicycle(1), Boat(2), Bottle(3), Bus(4), Car(5), Cat(6), Chair(7), Cup(8), Dog(9), Motorbike(10), People(11), Table(12)"
class_string = class_string.replace(" ", "")
classes = [x.split("(")[0] for x in class_string.split(",")]
print(classes)