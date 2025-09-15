#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cassert>
#include <algorithm>

class TradingAgent {
private:
    double balance;
    double position;  // Current stock position
    double max_position;
    double risk_limit;  // Maximum loss threshold
    std::vector<std::map<std::string, double>> trade_history;

public:
    TradingAgent(double initial_balance, double max_pos, double risk)
        : balance(initial_balance), position(0.0),
          max_position(max_pos), risk_limit(risk) {}

    std::string analyze_market(const std::vector<double>& price_data) {
        if (price_data.size() < 3) {
            return "HOLD";
        }

        double recent_trend = (price_data.back() - price_data[price_data.size() - 3])
                              / price_data[price_data.size() - 3];

        if (recent_trend > 0.02) {
            return "BUY";
        } else if (recent_trend < -0.02) {
            return "SELL";
        } else {
            return "HOLD";
        }
    }

    double calculate_position_size(const std::string& signal, double current_price) {
        if (signal == "HOLD") {
            return 0.0;
        }

        double available_capital = balance * 0.1;  // Use 10% of balance
        double position_size = available_capital / current_price;

        if (signal == "BUY") {
            double max_buy = max_position - position;
            return std::min(position_size, max_buy);
        } else {  // SELL
            return std::min(position_size, position);
        }
    }

    void execute_trade(const std::vector<double>& price_data) {
        double current_price = price_data.back();
        std::string signal = analyze_market(price_data);
        double trade_size = calculate_position_size(signal, current_price);

        if (std::fabs(trade_size) > 0.01) {  // Minimum trade threshold
            if (signal == "BUY") {
                double cost = trade_size * current_price;
                balance -= cost;
                position += trade_size;
            } else if (signal == "SELL") {
                double proceeds = trade_size * current_price;
                balance += proceeds;
                position -= trade_size;
            }

            std::map<std::string, double> trade = {
                {"action", signal == "BUY" ? 1.0 : (signal == "SELL" ? -1.0 : 0.0)},
                {"price", current_price},
                {"size", trade_size},
                {"balance", balance},
                {"position", position}
            };
            trade_history.push_back(trade);
        }
    }

    bool check_risk_limits(double current_price) {
        double portfolio_value = balance + (position * current_price);

        double initial_value = balance;
        for (const auto& trade : trade_history) {
            if (trade.at("action") == 1.0) {  // BUY
                initial_value += trade.at("size") * trade.at("price");
            }
        }

        if (initial_value > 0) {
            double current_loss = (initial_value - portfolio_value) / initial_value;
            return current_loss <= risk_limit;
        }
        return true;
    }

    void autonomous_trading_loop(const std::vector<double>& market_data, int steps) {
        for (int i = 0; i < std::min(steps, static_cast<int>(market_data.size()) - 2); ++i) {
            std::vector<double> current_window(
                market_data.begin() + i,
                market_data.begin() + i + 3
            );

            if (!check_risk_limits(current_window.back())) {
                std::cout << "Risk limit exceeded at step " << i << std::endl;
                break;
            }

            execute_trade(current_window);

            // Safety checks (verification properties)
            assert(balance >= 0.0 && "Balance cannot be negative");
            assert(std::fabs(position) <= max_position && "Position exceeds limits");
        }
    }

    double get_balance() const { return balance; }
    double get_position() const { return position; }
    const std::vector<std::map<std::string, double>>& get_trade_history() const {
        return trade_history;
    }
};

// Example usage
int main() {
    TradingAgent agent(10000.0, 100.0, 0.05);

    std::vector<double> market_prices = {100, 102, 105, 103, 108, 110, 107, 112, 115, 111};

    agent.autonomous_trading_loop(market_prices, 5);

    std::cout << "Final Balance: " << agent.get_balance() << std::endl;
    std::cout << "Final Position: " << agent.get_position() << std::endl;

    return 0;
}
#if 0
# Properties to verify:
# 1. SAFETY: Balance never goes negative
# 2. BOUNDS: Position never exceeds max_position
# 3. TERMINATION: Trading loop always terminates
# 4. RISK: Total loss never exceeds risk_limit
# 5. INVARIANT: Portfolio value remains within expected bounds
#endif
