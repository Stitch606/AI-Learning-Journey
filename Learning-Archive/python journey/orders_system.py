menu = {1:"hamburger",2:"Pizza",3:"Hotdog"}
price_list = {"hamburger":5,"Pizza":12,"Hotdog":4}
User_order = {}
print("wellcome")
print("menu: 1.hamburger 2.Pizza 3.Hotdog")
while True :
    try :
        option_1 = int(input("Enter your number orders :"))
        order = menu.get(option_1)
        if order :
            price = price_list.get(order)
            User_order.update({order:price})
            option_2 = input("Do you want to continue:(y/n)")
            if option_2 == "y" :
                continue
            elif option_2 ==  "n" :
                print(f"thank you{User_order}your orders")
                break
    except :
        print("incorrect try again")
total_price = sum(User_order.values())
print(f"your total price is : {total_price}")
