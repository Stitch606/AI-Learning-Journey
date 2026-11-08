print("wellcome")
#user input
data_list = [float(i) for i in input("enter data spreated by comma :").split(",")]
 #calculat mean
mean = sum(data_list)/len(data_list)
xi = [(i - mean) ** 2 for i in data_list]
variance = sum(xi)/len(data_list)
SD = round((variance ** 0.5),2)
mode = max(data_list, key=data_list.count)
print("mode :",mode)
print("mean :",mean)
print("SD :",SD)
