import quandl


def main():
    quandl.ApiConfig.api_key = "DigzLvJfs-nuxcSuZWuB"

    mydata = quandl.get('NSE/DRDATSONS')
    mydata.to_csv("NSE_DRDATSONS.csv")



if __name__ == '__main__':
    main()