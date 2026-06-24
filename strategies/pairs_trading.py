from core import spread, ou_fitting
from core import metrics
import pandas as pd


class OUpairTrader :
    def __init__(self,config):

        trading = config["trading"]
        ou      = config["ou"]

        self.M = trading["M"]               # rolling window (paper: 63)
        self.K = trading["K"]               # entry threshold (paper: 2.0)
        self.r = trading["stop_loss_r"]     # stop-loss fraction (paper: 0.1)

        g = ou["hedge_ratio_grid"]          # [min, max, step]
        self.g_min  = g[0]
        self.g_max  = g[1]
        self.g_step = g[2]

        # set during fit()
        self.b_ratio  = None
        self.mu       = None
        self.theta    = None
        self.sigma    = None
        self._s1_col  = None
        self._s2_col  = None

        # state reset before each run()
        self._position     = 0
        self._entry_spread = None



        pass

    def fit(self, data: pd.DataFrame):

        """
        data : output of fetch_pair_prices() for the formation period
               DatetimeIndex, columns ["T1_close", "T2_close"]
        """

        self._s1_col = data.columns[0]
        self._s2_col = data.columns[1]

        s1 = data[self._s1_col].values
        s2 = data[self._s2_col].values

        b_ratio, mu, theta, sigma = ou_fitting.fit_hedge_ratio(s1, s2, self.g_min, self.g_max, self.g_step)

        self.b_ratio = b_ratio
        self.mu      = mu
        self.theta   = theta
        self.sigma   = sigma

        self._position     = 0
        self._entry_spread = None
        self._spread_sl    = None


    def get_spread_price(self, date, history):
        s1 = history.iloc[-1, 0]
        s2 = history.iloc[-1, 1]
        return s1 - self.b_ratio * s2

    
    def generate_signals(self,date, history_data):

        """
        date    : pd.Timestamp  — current bar date (provided by engine, unused directly)
        history : pd.DataFrame  — price_data.loc[:date]  (all rows up to & including date)
                  columns: [_s1_col, _s2_col]
 
        returns : int  +1 (long spread)  |  -1 (short spread)  |  0 (flat / exit)
        """

        if len(history_data) < self.M :
            return 0
        
        s1            = history_data[self._s1_col]
        s2            = history_data[self._s2_col]
        spread_serie  = spread.construct_spread(s1,s2,self.b_ratio)
        spread_window = spread_serie.iloc[-self.M:]  # ← always exactly M days


        ma  = spread_window.mean()
        std = spread_window.std()

        current_s   = spread_serie.iloc[-1]
        current_ma  = ma.iloc[-1]
        current_std = std.iloc[-1] 
        
        if pd.isna(current_ma) or pd.isna(current_std) or current_std == 0:
            return 0
        
        upper = current_ma + self.K * current_std
        lower = current_ma - self.K * current_std
        


        #ENTRY SIGNALS 
        if self._position == 0 :
            if current_s < lower:

                self._position     = 1
                self._entry_spread = current_s
                self._spread_sl    = self._entry_spread * (1 - self.r)

                return 1

            elif current_s > upper :

                self._position     = -1
                self._entry_spread = current_s
                self._spread_sl    = self._entry_spread * (1 + self.r)

                return -1

        #EXIT SIGNALS

        elif self._position == 1:
            # exit long when spread crosses back above the moving average
            if current_s > current_ma:
                self._position    = 0
                self._entry_spread = None
                return 0

            # stop-loss: spread fell r% below entry  
            if self._entry_spread is not None and current_s < self._spread_sl:
                self._position    = 0
                self._entry_spread = None
                return 0

        elif self._position == -1:
            # exit short when spread crosses back below the moving average
            if current_s < current_ma:
                self._position    = 0
                self._entry_spread = None
                return 0

            # stop-loss: spread rose r% above entry  
            if self._entry_spread is not None and current_s > self._spread_sl:
                self._position    = 0
                self._entry_spread = None
                return 0
        
        return self._position
    
    def get_spread_price(self, date, history: pd.DataFrame) -> float:
        s1 = history[self._s1_col].iloc[-1]
        s2 = history[self._s2_col].iloc[-1]
        return float(s1 - self.b_ratio * s2)



        


        
        
        