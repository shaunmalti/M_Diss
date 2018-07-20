import quandl


def main():
    quandl.ApiConfig.api_key = "DigzLvJfs-nuxcSuZWuB"

    mydata = quandl.get('NASDAQOMX/XQC')
    mydata.to_csv("NASDAQOMX_XQC.csv")



if __name__ == '__main__':
    main()