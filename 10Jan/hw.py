class Employee:
    def __init__(self, employee_name, current_base_salary, tds_rate, da_rate):
        """The Employee constructor expects all params to be of the correct type.
        employee_name         -> string
        current_base_salary   -> int    (Annual salary)
        tds_rate              -> int
        da_rate               -> int

        If the provided tds_rate or da_rate are not in the range [0, 100],
        the closest integer from [0, 100] is used instead.
        """
        self.employee_name       = employee_name
        self.current_base_salary = current_base_salary
        self.tds_rate            = min(max(0, tds_rate), 100)
        self.da_rate             = min(max(0, da_rate), 100)

    def tds_amount(self):
        """Returns the final TDS amount in INR, rounded to the nearest integer."""
        return round(self.current_base_salary * self.tds_rate / 100)

    def da_amount(self):
        """Returns the DA amount in INR, rounded to the nearest integer."""
        return round(self.current_base_salary * self.da_rate / 100)

    def net_salary(self):
        """Returns the final salary in INR after adding DA and deducting TDS,
        rounded to the nearest integer.
        """
        return self.current_base_salary + self.da_amount() - self.tds_amount()

if __name__ == "__main__":
    empl_name   = input("Enter your name: ").strip()
    salary      = int(input("Current base salary: ").strip())
    tds_rate    = int(input("TDS rate: ").strip())
    da_rate     = int(input("DA rate: ").strip())

    new_empl    = Employee(empl_name, salary, tds_rate, da_rate)
    print("TDS Amount: ", new_empl.tds_amount())
    print("DA Amount: ", new_empl.da_amount())
    print("Net Salary: ", new_empl.net_salary())
