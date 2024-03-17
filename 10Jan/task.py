import re
import numpy

class Student:
    def __init__(self, first_name, last_name, age, roll_no):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.roll_no = roll_no

    def get_degree_using_age(self):
        if self.age <= 22:
            return "BTech"
        elif self.age <= 24:
            return "MSc"
        else:
            return "PhD"

    def get_degree_using_roll(self):
        if (re.search("^[A-Z]", self.roll_no) != None):
            return "PhD"
        elif (re.search("^\d{2}MA", self.roll_no) != None):
            return "MSc"
        else:
            return "BTech"


if __name__ == "__main__":
    name    = input("Enter your name: ").strip().split()
    name    += ["" for i in range(0, 2 - len(name))]
    age     = int(input("Enter your age: ").strip())
    roll_no = input("Enter your roll number: ").strip()

    me      = Student(name[0], name[1], age, roll_no)
    print("Based on age, you are a ", me.get_degree_using_age(), "student.")
    print("Based on roll number, you are a ", me.get_degree_using_roll(), "student.")

