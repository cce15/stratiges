# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone
from typing import Optional, List

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class WT(IStrategy):
    """
    This strategy is hand-crafted and made with love to help myself and my good frinds to
    achive our financial freedom.

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    """
    Th
    """
    minimal_roi = {
        "0": 100
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30
    # ===============================================================================
    # Strategy parameters
    over_sell_level = IntParameter(-65, -45, default=-53, space="buy", optimize=True)
    take_profit_precent = DecimalParameter(0.003, 0.03, decimals=3, default=0.005, space="buy", optimize=True)
    bigger_trend_respect = CategoricalParameter(['1h', '4h', '1d', False], default=False, space="buy", optimize=True)
    # DCA configration
    # Enable The DCA and safty odrers in the strategy
    # position_adjustment_enable = BooleanParameter(default=False, space="buy", optimize=True)
    position_adjustment_enable = True
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=-1, space="buy", optimize=True)
    max_dca_multiplier = DecimalParameter(2, 10, decimals=1, default=5.5, space="buy", optimize=True)
    drow_down_dca_precentage = DecimalParameter(-0.01, -0.3, decimals=2, default=-0.05, space="buy", optimize=True)
    safty_order_size_precntage = DecimalParameter(0.1, 3, decimals=1, default=0.5, space="buy", optimize=True)

    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.05, -0.99, decimals=3, name='stoploss')]

        def trailing_space() -> List[Dimension]:
            # All parameters here are mandatory, you can only modify their type or the range.
            return [
                # Fixed to true, if optimizing trailing_stop we assume to use trailing stop at all times.
                Categorical([True], name='trailing_stop'),

                SKDecimal(0.01, 0.50, decimals=3, name='trailing_stop_positive'),
                # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
                # so this intermediate parameter is used as the value of the difference between
                # them. The value of the 'trailing_stop_positive_offset' is constructed in the
                # generate_trailing_params() method.
                # This is similar to the hyperspace dimensions used for constructing the ROI tables.
                SKDecimal(0.001, 0.1, decimals=3, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]

        # Define a custom max_open_trades space
        def max_open_trades_space(self) -> List[Dimension]:
            return [
                Integer(-1, 10, name='max_open_trades'),
            ]

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "WT": {
                    'green': {'color': 'green'},
                    'red': {'color': 'red'},
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        ap = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        esa = ta.EMA(ap, 10)
        d = ta.EMA(abs(ap - esa), 10)
        ci = ((ap - esa) / (0.015 * d))
        tci = ta.EMA(ci, 21)
        green = tci
        red = ta.SMA(np.nan_to_num(green), 4)
        dataframe['green'], dataframe['red'] = green, red

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                # Signal: RSI crosses above buy_rsi
                    (qtpylib.crossed_above(dataframe['green'], dataframe['red'])) &
                    (dataframe['green'] <= self.over_sell_level.value) &  # Guard: tema below BB middle
                    (dataframe['red'] <= self.over_sell_level.value) &  # Guard: tema is raising
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'alhsan_cross')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        if current_profit > self.take_profit_precent.value:
            # if trade.nr_of_successful_buys >1:
            #     return f'DCA_EXIT : {trade.nr_of_successful_buys}'
            return 'Normal_Exit'

    # This is called when placing the initial order (opening trade)
    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: float, max_stake: float,
    #                         entry_tag: Optional[str], **kwargs) -> float:
    #
    #     # We need to leave most of the funds for possible further DCA orders
    #     # This also applies to fixed stakes
    #     return proposed_stake / self.max_dca_multiplier.value

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """

        if current_profit > self.drow_down_dca_precentage.value:
            return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = trade.nr_of_successful_buys
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_buys[0].cost
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_buys * self.safty_order_size_precntage.value))
            return stake_amount
        except Exception as exception:
            return None

        return None
