import numpy as np
import pandas as pd
import random
from sklearn.gaussian_process import GaussianProcessRegressor




from numpy.random import normal

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot as plt

class DCA():

    def __init__(self, start_date, end_date):

        self.btc_prices = pd.read_csv("BTC_NZD 2016-22.txt")

        
        self.fear_greed = pd.read_csv("crypto fear and greed.txt")

        self.btc_prices = self.btc_prices.drop(labels = range(0,4),axis=0)
        self.fear_greed = self.fear_greed.drop(labels = range(0,4),axis=0)

        self.btc_prices.Price = [elem.replace(',', '') for elem in self.btc_prices.Price]

        self.split_btc_date(end_date, start_date)

        self.fear_greed_values = self.fear_greed.values[:,1]

        self.fear_greed_sections = np.divide(self.fear_greed_values,10).astype(int)

        self.btc_buy_prices = self.btc_prices.iloc[::7, :]


        self.weekly_fear_greed = self.fear_greed_values[::7]

        

        self.buy_dict = {}


    def split_btc_date(self,start_date, end_date):

        start_idx = self.btc_prices.index
        start = start_idx[self.btc_prices.Date == start_date].tolist()[0]

        end_idx = self.btc_prices.index
        end = end_idx[self.btc_prices.Date == end_date].tolist()[0]

        self.btc_prices = self.btc_prices.loc[start:end]
        self.fear_greed = self.fear_greed.loc[start:end]


        return 

    def calculate_btc(self):

        btc_bought = 0
        total_spent = 0
        for i in range(len(self.btc_buy_prices)):

            key = int(self.weekly_fear_greed[i]/10)
            total_spent = total_spent + self.buy_dict[key]
            btc_bought = btc_bought + self.buy_dict[key] / float(self.btc_buy_prices.iloc[i].Price)

        avg_price = total_spent / btc_bought
        return avg_price, btc_bought, total_spent/len(self.weekly_fear_greed)


    


    # num_splits = 20

    # for i in range(num_splits):
    #     self.buy_dict[i] = 400 - i * 40

    
    # print('DCA with Fear-greed index')
    # print(f'Total spent: ${total_spent}')
    # print(f'Amount spent per week: {total_spent/len(self.btc_buy_prices)}')
    # print(f'BTC bought: {btc_bought}')
    # print(f'Average BTC cost price: {total_spent/btc_bought}')

    def random_search(self):

        
        avg_prices = []
        coeffs_lst = []
        for n in range(self.num_iters_RS):

            self.num_splits = 10
            spend_coeffs = sorted(random.choices(range(50,400),k=self.num_splits),reverse=True)
            

            for i in range(self.num_splits):
                self.buy_dict[i] = spend_coeffs[i]

            avg_prices.append(self.calculate_btc()[0])
            coeffs_lst.append(spend_coeffs)

        avg_prices = np.array(avg_prices)
        coeffs_lst = np.array(coeffs_lst)
        
        
        return avg_prices, coeffs_lst


    # surrogate or approximation for the objective function
    def surrogate(self,model,X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return model.predict(X)




 
    # probability of improvement acquisition function
    def acquisition(self,model,X,Y,num_samples):
        


        X_samples = np.array([sorted(random.choices(range(50,400),k=self.num_splits),reverse=True) for i in range(num_samples)])
        X_samples = X_samples.reshape(X_samples.shape[0],-1)

        
        best = min(Y)
        # calculate mean and stdev via surrogate function
        mu = self.surrogate(model, X_samples)
        
        # calculate the probability of improvement
        return X_samples[np.argmin(mu)], min(mu)
        
    



    def plot_price_split(self,X,Y,split_idx):

        plt.figure()

        plt.plot(X[:,split_idx],Y,'ko',markersize=8)
        plt.ylabel('BTC price (NZD)')
        plt.xlabel(f'Amount spent when fear greed index = {split_idx}')
        plt.title('Average BTC buy price vs amount bought when fear greed index = 0')
        plt.savefig(f'avg_buy_price_{split_idx}.png',dpi=300)

                

    def bayesian_opt(self, coeffs, avg_prices,year_str):

        

        X = coeffs
        Y = avg_prices


        count = X.shape[0]

        for i in range(self.num_iters_BO):

            count = count + 1
            #fitting model
            model = RandomForestRegressor()
            model.fit(X, Y)

            best_coeffs, best_price = self.acquisition(model,X,Y,num_samples=1000)
            X = np.append(X,best_coeffs).reshape(count,self.num_splits)
            Y = np.append(Y,best_price)
            print(best_price)

        # for idx in range(10):
        #     self.plot_price_split(X,Y,idx)

        Y, X = zip(*sorted(zip(list(Y), list(X))))

        X = np.array(X)
        Y = np.array(Y)

        np.savetxt(f"best_coeffs_{year_str}.csv", X[:self.num_return,:], delimiter=",")


        
    def calculate_coeffs_price(self,year_str):

        best_coeffs = np.genfromtxt(f"best_coeffs_{year_str}.csv", delimiter=",")

        prices = []
        bought = []
        spent = []

        for i in range(len(best_coeffs)):

            self.buy_dict = {}

            for j in range(self.num_splits):
                self.buy_dict[j] = best_coeffs[i][j]

            avg_price, btc_bought, total_spent = self.calculate_btc()
            prices.append(avg_price)
            bought.append(btc_bought)
            spent.append(total_spent)

        return prices, bought, spent


    def plot_btc_bought_vs_price(self):

        plt.figure()

        colors = ['ko', 'go', 'ro', 'bo']
        labels = ['2019-2020', '2020-2021', '2021-2022', '2019-2022']

        for key in self.year_dict:
            plt.plot(self.year_dict[key][1],self.year_dict[key][0],colors.pop(-1),ms=8, label=labels.pop(0))

        self.buy_dict = {i:200 for i in range(10)}

        avg_price, btc_bought, total_spent = self.calculate_btc()

        plt.plot(btc_bought, avg_price, 'mo', ms=8,label='$200')

        plt.ylabel('Average BTC price (NZD)')
        plt.xlabel('BTC bought')
        plt.legend()
        plt.savefig('btc_bought_vs_price.png', dpi=300)


        plt.figure()

        colors = ['ko', 'go', 'ro', 'bo','yo']
        labels = ['2018-2019','2019-2020', '2020-2021', '2021-2022', '2019-2022']

        for key in self.year_dict:
            plt.plot(self.year_dict[key][1],self.year_dict[key][2],colors.pop(-1),ms=8, label=labels.pop(0))

        self.buy_dict = {i:200 for i in range(10)}

        avg_price, btc_bought, total_spent = self.calculate_btc()

        plt.plot(btc_bought, total_spent, 'mo', ms=8,label='$200')

        plt.ylabel('Total money spent per week(NZD)')
        plt.xlabel('BTC bought')
        plt.legend()
        plt.savefig('btc_bought_vs_spent.png', dpi=300)


    def plot_fear_greed_histogram(self):

        plt.figure()
        plt.hist(self.fear_greed_sections,bins=10)

        plt.ylabel('Section frequency')
        plt.xlabel('Section')

        plt.savefig('Section_histogram.png',dpi=300)
        





    





            





            






        print(2)







    


if __name__ == "__main__":

    dates = ["Feb 01, 2018", "Jan 31, 2019", "Jan 30, 2020", "Jan 28, 2021", "Mar 03, 2022"]

    for i in range(len(dates)-1):

        btc_dca = DCA(dates[i], dates[i+1])

        btc_dca.num_iters_RS = 200
        avg_prices, coeffs_lst = btc_dca.random_search()

        btc_dca.num_return = 25
        btc_dca.num_iters_BO = 10

        year = dates[i][-4:]
        year_str = f"{year}-{int(year)+1}"

        btc_dca.bayesian_opt(coeffs_lst, avg_prices, year_str)

    #3 year period between 2019-2022
    btc_dca = DCA(dates[0], dates[-1])
    btc_dca.num_iters_RS = 200
    btc_dca.num_return = 25
    btc_dca.num_iters_BO = 10
    avg_prices, coeffs_lst = btc_dca.random_search()

    year = dates[0][-4:]
    year_str_long = f"{year}-{int(year)+4}"

    btc_dca.bayesian_opt(coeffs_lst, avg_prices, year_str_long)

    btc_dca.year_dict = {}

    for i in range(len(dates)-1):

        year = dates[i][-4:]
        year_str = f"{year}-{int(year)+1}"

        prices, bought, spent = btc_dca.calculate_coeffs_price(year_str)

        btc_dca.year_dict[year_str] = [prices,bought, spent]

    prices, bought, spent = btc_dca.calculate_coeffs_price(year_str_long)

    btc_dca.year_dict[year_str_long] = [prices,bought,spent]

    btc_dca.plot_btc_bought_vs_price()

    btc_dca.plot_fear_greed_histogram()


    

    


    







        
    









    print(2)


    #TO DO: write function to grab data from past 3/4/5 years

    #Write optimisation to optimise amount bought for 10 intervals (e.g. $100 if fear_greed > 90 etc)


    #TO DO:

    # *Write a function to extract only wanted dates for BTC
    # *Bayesian optimisation