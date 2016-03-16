
# porfolio-optimization

1. Requirements 
 * gurobi 6.5.0 (License is needed)
 * python 2.7:
 * Flask 0.10.1
 * Numpy 1.9.2
 * Pandas 0.16.2
 
2. Get started
 * Run "python index.py" in terminal.
 * Open 127.0.0.1:12345 in a browser.
 * Select start- and end-date; add more tickers.
 * Click "Quote" to process data in the data folder, return a oprtimized portfolio
 * Change the portfolio by varying the options provided until u get the best combination
 * Replace the data files with your own data to include more instruments and take more time period

3. Yet to be done
 * Integrate backend transaction costs with frontend
 * Include Tax constraints if the investor is looking for huge amounts, for eg > 10m
 * Connect to database which updates data on a realtime basis
 * Include short sell strategies
 * Introduce parallelism to the server
