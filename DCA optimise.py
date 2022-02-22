import numpy as np
import pandas as pd

btc_prices = pd.read_csv("BTC_NZD 2019-22.txt")
fear_greed = pd.read_csv("crypto fear and greed.txt")

btc_prices = btc_prices.drop(labels = range(0,5),axis=0)
fear_greed = fear_greed.drop(labels = range(0,5),axis=0)

btc_prices.Price = [elem.replace(',', '') for elem in btc_prices.Price]

fear_greed_values = fear_greed.values[:,1]

btc_buy_prices = btc_prices.iloc[::7, :]

#STEPS

# Go thru in steps of 7 (Wednesdays)
# Get average value from previous week
# 

#

weekly_fear_greed = np.array_split(fear_greed_values,int(len(fear_greed_values)/7))

avg_weekly_fear_greed = [np.mean(elem) for elem in weekly_fear_greed]

buy_dict = {}

num_splits = 20

for i in range(num_splits):
    buy_dict[i] = 300 - i * 10

btc_bought = 0
total_spent = 0
for i in range(len(btc_buy_prices)):

    key = int(avg_weekly_fear_greed[i]/10)
    total_spent = total_spent + buy_dict[key]
    btc_bought = btc_bought + buy_dict[key] / float(btc_buy_prices.iloc[i].Price)

print('DCA with Fear-greed index')
print(f'Total spent: ${total_spent}')
print(f'Amount spent per week: {total_spent/len(btc_buy_prices)}')
print(f'BTC bought: {btc_bought}')
print(f'Average BTC cost price: {total_spent/btc_bought}')


print('DCA blindly')
btc_bought = 0
total_spent = 0

amount_spent = 200
for i in range(len(btc_buy_prices)):

    
    total_spent = total_spent + amount_spent
    btc_bought = btc_bought + amount_spent / float(btc_buy_prices.iloc[i].Price)

print(f'Total spent: ${total_spent}')
print(f'Amount spent per week: {total_spent/len(btc_buy_prices)}')
print(f'BTC bought: {btc_bought}')
print(f'Average BTC cost price: {total_spent/btc_bought}')






print(2)


#TO DO: write function to grab data from past 3/4/5 years

#Write optimisation to optimise amount bought for 10 intervals (e.g. $100 if fear_greed > 90 etc)