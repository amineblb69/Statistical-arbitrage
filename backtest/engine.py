import pandas as pd

class BacktestEngine:

    """
    Simulates the two-leg spread position exactly as described in the paper (p.20-21).
 
    Position mechanics (paper Step 3, p.20):
    ─────────────────────────────────────────
    Spread :  Xt  = S1t  -  B · S2t
 
    LONG ENTRY  (Xt < MA - K·SD):
        qty       =  Ct / S1t          — invest ALL cash into S1
        leg_s1    = +qty shares of S1  (long)
        leg_s2    = -B·qty shares of S2 (short)
        Ct        =  Ct - Xt · qty     — paper's cash update
 
    SHORT ENTRY  (Xt > MA + K·SD):
        qty       = -Ct / S1t          — negative qty
        leg_s1    = +qty  (negative → short S1)
        leg_s2    = -B·qty (negative→negative → long S2)
        Ct        =  Ct - Xt · qty
 
    EXIT / STOP-LOSS:
        PnL from S1 leg : (S1t_exit - S1t_entry) · leg_s1_qty
        PnL from S2 leg : (S2t_exit - S2t_entry) · leg_s2_qty
        Total PnL       = pnl_s1 + pnl_s2
        Ct              = Ct + Xt · qty   (paper's cash update on exit)
    """

    def __init__(self, initial_capital: float, config: dict):
        
        self.initial_capital = initial_capital
        self.config          = config
        self.capital         = initial_capital
        self.positions       = {}
        self.trade_log       = []
        self.pnl_log         = []

    def run(self, pair_id: list, strategy, price_data: pd.DataFrame):

        """
        pair_id    : list           e.g. ["NVDA","AMD"]
        strategy   : OUPairTrader  already fitted on formation data
        price_data : pd.DataFrame  trading-period prices from fetch_pair_prices()
                     DatetimeIndex, columns ["T1_close", "T2_close"]
 
        returns    : pd.DataFrame  columns ["date", "pnl", "capital"]
        """
        strategy._position     = 0       # ← ADD THESE
        strategy._entry_spread = None
        strategy._spread_sl    = None

        self.capital   = self.initial_capital
        self.positions = {}
        self.pnl_log   = []

        s1_col = price_data.columns[0]
        s2_col = price_data.columns[1]

        for date, _ in price_data.iterrows():

            history = price_data.loc[:date]

            s1_cprice = float(history[s1_col].iloc[-1])
            s2_cprice = float(history[s2_col].iloc[-1])

            signal       = strategy.generate_signals(date, history)
            spread_price = strategy.get_spread_price(date, history)
            pnl          = self._execute_signal(
                signal      = signal,
                spread      = spread_price,
                s1_price    = s1_cprice,
                s2_price    = s2_cprice,
                b_ratio     = strategy.b_ratio,
                capital     = self.capital,
                pair        = pair_id,
                date        = date,
            )

            self.capital += pnl    
            self.pnl_log.append({'date': date, 'pnl': pnl, 'capital': self.capital})    

        return pd.DataFrame(self.pnl_log)

    def _execute_signal(
        self,
        signal   : int,    # +1 | -1 | 0  from generate_signal()
        spread   : float,  # Xt = S1t - B·S2t
        s1_price : float,  # S1t  (raw asset price, not spread)
        s2_price : float,  # S2t  (raw asset price, not spread)
        b_ratio  : float,  # B    (hedge ratio from fit())
        capital  : float,  # Ct   (current cash)
        pair     : list,
        date,
        ) -> float:

        date       = pd.to_datetime(date)
        trading    = self.config.get("trading",{})
        commission = self.config.get('commission', 0.0)
        slippage   = self.config.get('slippage',   0.0)
        
        prev_signal   = self.positions.get("signal", 0)
        prev_qty      = self.positions.get("qty", 0.0)
        prev_s1_entry = self.positions.get("s1_entry", s1_price)
        prev_s2_entry = self.positions.get("s2_entry", s2_price)
        prev_date     = self.positions.get("date", date) 

        net_pnl = 0.0


        #CLOSING POSITION :

        if signal == 0 and signal != prev_signal :

            #Apply slippage against the direction we are closing
            s1_exit = s1_price * (1 - prev_signal * slippage)
            s2_exit = s2_price * (1 + prev_signal * slippage)

            #Get the PnL of each leg individually 
            #leg1   qty = + prev_qty
            #leg_s2 qty = -B · prev_qty
            s1_pnl = (s1_exit - prev_s1_entry) * prev_qty
            s2_pnl = (s2_exit - prev_s2_entry) * (-b_ratio * prev_qty)

            prev_capital = self.positions.get("entry_capital", capital)

            cost_s1 = abs(prev_qty          * s1_exit) * commission
            cost_s2 = abs(b_ratio * prev_qty * s2_exit) * commission

            net_pnl = s1_pnl + s2_pnl - cost_s1 -cost_s2

            prev_entry_spread = self.positions.get("entry_spread", spread)

            self.trade_log.append({
                'pair'         : pair,
                'side'         :'long_spread' if prev_signal == 1 else 'short_spread',
                'entry_date'   : prev_date,
                'exit_date'    : date,
                's1_entry'     : prev_s1_entry,
                's2_entry'     : prev_s2_entry,
                's1_exit'      : s1_exit,
                's2_exit'      : s2_exit,
                'entry_spread' : prev_entry_spread,
                'exit_spread'  : spread,
                'qty'          : prev_qty,
                's1_pnl'     : s1_pnl,
                's2_pnl'     : s2_pnl,
                's1_cost'    : cost_s1,
                's2_cost'    : cost_s2,
                'total_cost' : cost_s1 + cost_s2,
                'net_pnl'    : net_pnl,
                'capital'    : prev_capital + net_pnl,
                'return'     : net_pnl / prev_capital * 100 if prev_capital !=0 else 0.0,                            
                })
            
            self.positions = {}


        #ENTER POSITION : 
        if signal != 0 and prev_signal == 0 :
        

            #Apply slippage against direction
            s1_fill = s1_price * (1 + signal * slippage)
            s2_fill = s2_price * (1 - signal * slippage)

            #signal = +1 → qty positive → long S1, short S2
            #signal = -1 → qty negative → short S1, long S2
            qty = signal * capital / s1_fill

            # entry costs on both legs
            cost_s1 = abs(qty          * s1_fill) * commission
            cost_s2 = abs(b_ratio * qty * s2_fill) * commission

            self.positions = {
                'signal'        : signal,
                'qty'           : qty,          
                's1_entry'      : s1_fill,
                's2_entry'      : s2_fill,
                'entry_spread'  : spread,
                'date'          : date,
                'entry_cost'    : cost_s1 + cost_s2,
                'entry_capital' : capital
            }

            net_pnl -= (cost_s1 + cost_s2)

        return net_pnl
    
    def mark_to_market(self, current_spread : float) -> float :

        """
        The goal is to return the current total equity : available cash + unrealised pnl 
        to make a good rebalance based on all the equity not only the availablecash.
        Paper's formula : C-t + X_t * POS(X_t)
        """

        if not self.positions:
            return self.capital
        
        pos          = self.positions.get('qty',0.0)
        entry_spread = self.positions.get('entry_spread',current_spread)
        unrealised   = (current_spread - entry_spread) * pos 

        return self.capital + unrealised