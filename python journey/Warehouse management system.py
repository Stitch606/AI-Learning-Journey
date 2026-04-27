from datetime import datetime

class Product_Management :
    def __init__(self,product,price,quantity) :
        self.product = product
        self.price = price
        self.quantity = quantity
    def manage(self) :
        products.append(self)
class sell :
    def __init__(self,solled_product,solled_quantity) :
        self.solled_product = solled_product
        self.solled_quantity = solled_quantity
        self.sale_time = datetime.now()
    def add_salles(self) :
        sales_operations.append(self) 

products = []

sales_operations = []

print("wellcom")

main = int(input("1 for sell 2 for manage :"))

if main == 1 :
    for i in products :
        print(i.product,":",i.price,i.quantity)
    while True :
        print("1.remove product")
        print("2.new price")
        print("3.add product")
        print("4.Sales operations")
        change = int(input("...."))
        match change :
            case 1 :
                target_name = input("name")
                for item in products:
                    if item.product == target_name :
                        products.remove(item) 
                        print("deleted")
                        break
            case 2 :
                name_product = input("name...")
                for item in products :
                    if item.product == name_product :
                        new_price = float(input("add new price :"))
                        item.price = new_price
                        print("price updated")
                        break
            case 3 :
                add_product = input("product name")
                the_price = float(input("price..."))
                quantity = int(input("quantity..."))
                new_product = Product_Management(add_product,the_price,quantity)
                new_product.manage()
            case 4 :
                for i in sales_operations :
                    print(i.solled_product,i.solled_quantity,i.sale_time)
            
                
elif main == 2 :
    while True :
        sold_product = input("Enter the name of the product sold")
        quantity_sold = int(input("Enter the quantity sold"))
        for i in products :
            if sold_product == i.product and i.price > 0 :
                print(i.price)
                if  i.quantity >= quantity_sold :
                    i.quantity -= quantity_sold
                    sales_record = sell(sold_product,quantity_sold)
                    sales_record.add_salles()
                    print("Sold")
                    break
                else :
                    print("Quantity not available")
                    break 
           else :
               for i in products :
                   print("Review the products ")
                   print(f"{i.product}, : {i.price}, : {i.quantity}")
             
                
                
                
                
                