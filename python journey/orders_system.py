Orders = []
print("wellcome")
print("menu 1.hamburger 2.Pizza 3.Hotdog")
while True :
    demand = int(input("Enter your orders:"))
    match demand :
        case 1:
            Orders.append("hamburger")
        case 2:
            Orders.append("Pizza")
        case 3:
            Orders.append("Hotdog")
    addition = input("Would you like to add anything else? yes/no :")
    if addition == "yes" :
        continue
    elif addition == "no" :

        print(f"your orders{Orders}")
        print("thank you")
        break