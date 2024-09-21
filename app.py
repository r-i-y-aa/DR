# print("hello world")
# patientName = "John Smith"
# patientAge = 20
# isNewPatient = False
# patientString = ""
# if isNewPatient == True:
#     patientString = patientName + " is a " + str(patientAge) + " year old patient." + " He is a new patient."
# else:
#     patientString = patientName + "is a " + str(patientAge) + " year old patient." + " He is not a new patient."
# print(patientString)
# name = input("What is your name? ")
# print("Hello " + name)
# birthYear = input("Enter your birth year: ")
# age = 2024 - int(birthYear)
# print("You are " + str(age) + " years old")
# firstNumber = input("First: ")
# secondNumber = input("Second: ")
# sum = float(firstNumber) + float(secondNumber)
# print("Sum: " + str(sum))
# course = 'Python for beginners'
# print(str('Python' in course))
weight = input("Weight: ")
weight_type = input("(K)g or (L)bs: ")
final_weight = 1;
if (weight_type.upper() == "L"):
    final_weight = float(weight)*0.45
    print("Weight in kg: " + str(final_weight))
elif(weight_type.upper() == "K"):
    final_weight = float(weight)*2.2
    print("Weight in lb: " + str(final_weight))
else:
    print("erm sry couldn't get that -_-")

