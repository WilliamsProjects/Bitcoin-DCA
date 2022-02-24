import numpy as np
import pandas as pd
import random

class DCA():

    def __init__(self):

        self.btc_prices = pd.read_csv("BTC_NZD 2019-22.txt")
        self.fear_greed = pd.read_csv("crypto fear and greed.txt")

        self.btc_prices = self.btc_prices.drop(labels = range(0,5),axis=0)
        self.fear_greed = self.fear_greed.drop(labels = range(0,5),axis=0)

        self.btc_prices.Price = [elem.replace(',', '') for elem in self.btc_prices.Price]

        self.fear_greed_values = self.fear_greed.values[:,1]

        self.btc_buy_prices = self.btc_prices.iloc[::7, :]


        self.weekly_fear_greed = np.array_split(self.fear_greed_values,int(len(self.fear_greed_values)/7))

        self.avg_weekly_fear_greed = [np.mean(elem) for elem in self.weekly_fear_greed]

        self.buy_dict = {}


    def calculate_btc(self):

        btc_bought = 0
        total_spent = 0
        for i in range(len(self.btc_buy_prices)):

            key = int(self.avg_weekly_fear_greed[i]/10)
            total_spent = total_spent + self.buy_dict[key]
            btc_bought = btc_bought + self.buy_dict[key] / float(self.btc_buy_prices.iloc[i].Price)

        avg_price = total_spent / btc_bought
        return avg_price


    


    # num_splits = 20

    # for i in range(num_splits):
    #     self.buy_dict[i] = 400 - i * 40

    
    # print('DCA with Fear-greed index')
    # print(f'Total spent: ${total_spent}')
    # print(f'Amount spent per week: {total_spent/len(self.btc_buy_prices)}')
    # print(f'BTC bought: {btc_bought}')
    # print(f'Average BTC cost price: {total_spent/btc_bought}')

    def random_search(self):

        num_iters = 100
        avg_prices = []
        coeffs_lst = []
        for n in range(num_iters):

            num_splits = 10
            spend_coeffs = sorted(random.choices(range(400),k=num_splits),reverse=True)

            for i in range(num_splits):
                self.buy_dict[i] = spend_coeffs[i]

            avg_prices.append(self.calculate_btc())
            coeffs_lst.append(spend_coeffs)
        
        return avg_prices, coeffs_lst




    


if __name__ == "__main__":

    btc_dca = DCA()

    avg_prices, coeffs_lst = btc_dca.random_search()








    print(2)


    #TO DO: write function to grab data from past 3/4/5 years

    #Write optimisation to optimise amount bought for 10 intervals (e.g. $100 if fear_greed > 90 etc)


    #TO DO:

    # *Write a function to extract only wanted dates for BTC
    # *Bayesian optimisation