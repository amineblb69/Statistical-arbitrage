import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime

from core.metrics import (
    compute_metrics,
    compute_metrics_from_equity,
    equity_returns,
    equity_drawdown,
)
from strategies.multipair_framework import MultiPairFramework
from data.loader import fetch_pair_prices

class PortfolioReport:

    def __init__(self, config, dict, pairs, initial_capital : float = 100_000.0):

        self.config = config   
        self.pairs  = [tuple(p)for p in pairs]
        self.initial_capital = initial_capital
        
        data_cfg = config.get("data",{})
        self.formation_start = data_cfg.get("fomation_start","2022-12-31")
        self.formation_end   = data_cfg.get("formation_end","")
