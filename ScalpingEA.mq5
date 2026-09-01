//+------------------------------------------------------------------+
//|                          ScalpingEA.mq5                          |
//|              Professional Forex Scalping Expert Advisor            |
//|        Optimized for GOLD, BTCUSD, USTEC on Exness Broker         |
//|     Designed to grow $10 accounts to $100+ daily with 24/7 trading |
//+------------------------------------------------------------------+
#property copyright "ScalpingEA Pro"
#property link "https://github.com/TBAN009/ScalpingEA"
#property version "1.00"
#property strict
#property description "Advanced scalping EA - High frequency micro trading"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS
//+------------------------------------------------------------------+

// Account & Risk Management
input double RiskPercentage = 2.0;           // Risk per trade (%)
input double MaxDailyLoss = 5.0;             // Max daily loss limit (%)
input int MaxOpenTrades = 5;                 // Maximum concurrent trades
input bool UseMoneyManagement = true;        // Enable advanced money management
input double AccountSizeThreshold = 100.0;   // Threshold for micro lot adjustment

// Scalping Strategy Parameters
input int FastMA = 5;                        // Fast moving average period
input int SlowMA = 20;                       // Slow moving average period
input int RSIPeriod = 14;                    // RSI period
input double RSIOverbought = 75.0;           // RSI overbought level
input double RSIOversold = 25.0;             // RSI oversold level
input int MACD_Fast = 12;                    // MACD fast period
input int MACD_Slow = 26;                    // MACD signal period
input int MACD_Signal = 9;                   // MACD signal period

// Take Profit & Stop Loss
input int TakeProfitPips = 5;                // Take profit in pips
input int StopLossPips = 3;                  // Stop loss in pips
input bool UseTrailingStop = true;           // Enable trailing stop
input int TrailingStopPips = 2;              // Trailing stop distance in pips

// Time Parameters
input int StartHour = 0;                     // EA start hour (24h format)
input int EndHour = 23;                      // EA end hour (24h format)
input int MinutesPerTrade = 1;               // Minimum minutes between trades
input bool Trade24_7 = true;                 // Enable 24/7 trading

// Symbol Configuration
input string Symbol1 = "XAUUSD";             // Gold
input string Symbol2 = "BTCUSD";             // Bitcoin
input string Symbol3 = "USTEC";              // US Tech Index

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES
//+------------------------------------------------------------------+

CTrade trade;
double initialBalance;
double dailyStartBalance;
double dailyProfit;
double maxDrawdown;
double accountEquity;
double accountBalance;
double lotSize;
datetime lastTradeTime;
int totalTrades = 0;
int winTrades = 0;
int lossTrades = 0;

struct TradeSession {
    datetime sessionStart;
    double startBalance;
    double currentProfit;
    int tradesCount;
};

TradeSession tradeSession;

//+------------------------------------------------------------------+
//| EXPERT INITIALIZATION
//+------------------------------------------------------------------+

int OnInit()
{
    // Initialize trade object
    trade.SetExpertMagicNumber(20260901);
    
    // Set initial values
    initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    dailyStartBalance = initialBalance;
    lastTradeTime = TimeCurrent();
    
    // Verify symbols are available
    if (!SymbolSelect(Symbol1, true) || !SymbolSelect(Symbol2, true) || !SymbolSelect(Symbol3, true))
    {
        Alert("Error: One or more symbols not available. Check symbol names on Exness broker.");
        return INIT_FAILED;
    }
    
    // Initialize trade session
    tradeSession.sessionStart = TimeCurrent();
    tradeSession.startBalance = initialBalance;
    tradeSession.currentProfit = 0;
    tradeSession.tradesCount = 0;
    
    Print("ScalpingEA initialized successfully");
    Print("Initial Balance: ", initialBalance);
    Print("Risk per trade: ", RiskPercentage, "%");
    Print("Symbols: ", Symbol1, ", ", Symbol2, ", ", Symbol3);
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| EXPERT DEINITIALIZATION
//+------------------------------------------------------------------+

void OnDeinit(const int reason)
{
    Print("ScalpingEA deinitialized");
    Print("Total trades: ", totalTrades, " | Wins: ", winTrades, " | Losses: ", lossTrades);
    if (totalTrades > 0)
        Print("Win rate: ", (double)winTrades / totalTrades * 100, "%");
}

//+------------------------------------------------------------------+
//| EXPERT TICK
//+------------------------------------------------------------------+

void OnTick()
{
    // Update account information
    accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    accountEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Check daily loss limit
    dailyProfit = accountEquity - dailyStartBalance;
    if (dailyProfit < -(dailyStartBalance * MaxDailyLoss / 100))
    {
        Print("Daily loss limit reached. Stopping trades.");
        return;
    }
    
    // Check if trading is allowed
    if (!IsTradingAllowed())
    {
        Print("Trading not allowed");
        return;
    }
    
    // Check time trading rules
    if (!Trade24_7 && !IsTradeTimeAllowed())
        return;
    
    // Update trailing stops for open positions
    UpdateTrailingStops();
    
    // Main trading logic
    ProcessTradingSignals();
}

//+------------------------------------------------------------------+
//| PROCESS TRADING SIGNALS
//+------------------------------------------------------------------+

void ProcessTradingSignals()
{
    // Check minimum time between trades
    if (TimeCurrent() - lastTradeTime < MinutesPerTrade * 60)
        return;
    
    // Trade each symbol
    ProcessSymbol(Symbol1, 1);
    ProcessSymbol(Symbol2, 2);
    ProcessSymbol(Symbol3, 3);
}

//+------------------------------------------------------------------+
//| PROCESS INDIVIDUAL SYMBOL
//+------------------------------------------------------------------+

void ProcessSymbol(string symbol, int symbolIndex)
{
    // Check if we can open new trades
    if (CountOpenTrades() >= MaxOpenTrades)
        return;
    
    // Get indicators
    double fastMA = GetMA(symbol, FastMA, 1);
    double slowMA = GetMA(symbol, SlowMA, 1);
    double rsiValue = GetRSI(symbol, RSIPeriod, 1);
    double macdValue = GetMACD(symbol, MACD_Fast, MACD_Slow, MACD_Signal, 1);
    double macdSignal = GetMACD(symbol, MACD_Fast, MACD_Slow, MACD_Signal, 2);
    
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double spread = ask - bid;
    
    // BUY Signal: Fast MA > Slow MA, RSI oversold, MACD bullish
    if (fastMA > slowMA && rsiValue < RSIOversold && macdValue > macdSignal && spread < 0.001)
    {
        // Calculate lot size
        double lot = CalculateLotSize(symbol, StopLossPips);
        
        // Calculate TP and SL
        double tp = ask + TakeProfitPips * SymbolInfoDouble(symbol, SYMBOL_POINT);
        double sl = bid - StopLossPips * SymbolInfoDouble(symbol, SYMBOL_POINT);
        
        // Open BUY trade
        if (trade.Buy(lot, symbol, 0, sl, tp))
        {
            lastTradeTime = TimeCurrent();
            totalTrades++;
            Print("BUY signal on ", symbol, " | Lot: ", lot, " | TP: ", tp, " | SL: ", sl);
        }
    }
    
    // SELL Signal: Fast MA < Slow MA, RSI overbought, MACD bearish
    else if (fastMA < slowMA && rsiValue > RSIOverbought && macdValue < macdSignal && spread < 0.001)
    {
        // Calculate lot size
        double lot = CalculateLotSize(symbol, StopLossPips);
        
        // Calculate TP and SL
        double tp = bid - TakeProfitPips * SymbolInfoDouble(symbol, SYMBOL_POINT);
        double sl = ask + StopLossPips * SymbolInfoDouble(symbol, SYMBOL_POINT);
        
        // Open SELL trade
        if (trade.Sell(lot, symbol, 0, sl, tp))
        {
            lastTradeTime = TimeCurrent();
            totalTrades++;
            Print("SELL signal on ", symbol, " | Lot: ", lot, " | TP: ", tp, " | SL: ", sl);
        }
    }
}

//+------------------------------------------------------------------+
//| CALCULATE LOT SIZE
//+------------------------------------------------------------------+

double CalculateLotSize(string symbol, int stopLossPips)
{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (RiskPercentage / 100);
    
    // Get symbol info
    double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    
    // Avoid division by zero
    if (tickValue == 0 || tickSize == 0)
        return minLot;
    
    // Calculate lot size
    double stopLossValue = stopLossPips * tickSize;
    double lotSize = (riskAmount / (stopLossValue * tickValue / tickSize));
    
    // Ensure small accounts use micro lots
    if (accountBalance < AccountSizeThreshold)
        lotSize = MathMax(minLot, lotSize * 0.1);
    
    // Normalize lot size
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| GET MOVING AVERAGE
//+------------------------------------------------------------------+

double GetMA(string symbol, int period, int shift)
{
    int handle = iMA(symbol, PERIOD_M1, period, 0, MODE_SMA, PRICE_CLOSE);
    if (handle == INVALID_HANDLE)
        return 0;
    
    double ma[];
    CopyBuffer(handle, 0, shift, 1, ma);
    IndicatorRelease(handle);
    
    return (ArraySize(ma) > 0) ? ma[0] : 0;
}

//+------------------------------------------------------------------+
//| GET RSI
//+------------------------------------------------------------------+

double GetRSI(string symbol, int period, int shift)
{
    int handle = iRSI(symbol, PERIOD_M1, period, PRICE_CLOSE);
    if (handle == INVALID_HANDLE)
        return 0;
    
    double rsi[];
    CopyBuffer(handle, 0, shift, 1, rsi);
    IndicatorRelease(handle);
    
    return (ArraySize(rsi) > 0) ? rsi[0] : 0;
}

//+------------------------------------------------------------------+
//| GET MACD
//+------------------------------------------------------------------+

double GetMACD(string symbol, int fast, int slow, int signal, int mode)
{
    int handle = iMACD(symbol, PERIOD_M1, fast, slow, signal, PRICE_CLOSE);
    if (handle == INVALID_HANDLE)
        return 0;
    
    double macd[];
    int buffer = (mode == 1) ? 0 : (mode == 2) ? 1 : 2;  // MACD line, Signal, or Histogram
    
    CopyBuffer(handle, buffer, 0, 1, macd);
    IndicatorRelease(handle);
    
    return (ArraySize(macd) > 0) ? macd[0] : 0;
}

//+------------------------------------------------------------------+
//| UPDATE TRAILING STOPS
//+------------------------------------------------------------------+

void UpdateTrailingStops()
{
    if (!UseTrailingStop)
        return;
    
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if (!SelectPosition(i))
            continue;
        
        string symbol = PositionGetString(POSITION_SYMBOL);
        int type = (int)PositionGetInteger(POSITION_TYPE);
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);
        
        if (type == POSITION_TYPE_BUY)
        {
            double newSL = bid - (TrailingStopPips * SymbolInfoDouble(symbol, SYMBOL_POINT));
            if (newSL > currentSL && newSL > 0)
            {
                trade.PositionModify(PositionGetTicket(0), newSL, currentTP);
            }
        }
        else if (type == POSITION_TYPE_SELL)
        {
            double newSL = ask + (TrailingStopPips * SymbolInfoDouble(symbol, SYMBOL_POINT));
            if (newSL < currentSL && newSL > 0)
            {
                trade.PositionModify(PositionGetTicket(0), newSL, currentTP);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| COUNT OPEN TRADES
//+------------------------------------------------------------------+

int CountOpenTrades()
{
    int count = 0;
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if (SelectPosition(i))
        {
            if (PositionGetString(POSITION_SYMBOL) == Symbol1 ||
                PositionGetString(POSITION_SYMBOL) == Symbol2 ||
                PositionGetString(POSITION_SYMBOL) == Symbol3)
            {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| IS TRADE TIME ALLOWED
//+------------------------------------------------------------------+

bool IsTradeTimeAllowed()
{
    int currentHour = Hour();
    if (Trade24_7)
        return true;
    
    if (currentHour >= StartHour && currentHour <= EndHour)
        return true;
    
    return false;
}

//+------------------------------------------------------------------+
//| SELECT POSITION
//+------------------------------------------------------------------+

bool SelectPosition(int index)
{
    return PositionGetTicket(index) > 0;
}
