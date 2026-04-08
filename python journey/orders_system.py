Orders = {1:"hamburger",2:"Pizza",3:"Hotdog"}
price_list = {"hamburger":5,"Pizza":12,"Hotdog":4}
User_order = {}
print("wellcome")
print("menu: 1.hamburger 2.Pizza 3.Hotdog")
while True :
    option_1 = (input("Enter your number orders or q to exit:"))
    order = Orders.get(option_1)
    if order :
        price = price_list.get(order)
        User_order.update({order:price})
        option_2 = input("Do you want to continue (y/n):")
    if "y" :
        continue
    elif "n" :
        print(f"thank you{User_order}your orders")
        
        
         
    
    
