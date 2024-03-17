# task 1
def numNonBinary(arr):
    result = 0
    for element in arr:
        if element < 0 or element > 1:
            result = result + 1
    return result

def is2024Present(*args):
    return 2024 in args

def getCitiesAndCost(**cityCost):
    cities = []
    totalCost = 0

    for city, cost in cityCost.items():
        totalCost += cost
        cities.append(city)

    return cities, totalCost

if __name__ == "__main__":
    # task1
    # arr = []
    # num = int(input("Enter the number of elements: "))
    # for i in range(0, num):
    #     elem = float(input(">> "))
    #     arr.append(elem)
    # print("Result: " + str(numNonBinary(arr)))

    # task 2
    # print("Result :" + str(is2024Present(1, 2, 69, 120, 2024)))

    # homework
    cities, cost = getCitiesAndCost(agra=20, itanagar=15, bhubaneswar=10)
    print("Cities: ", cities)
    print("Cost: ", cost)
