// Hackathon Starter: Complete AI Financial Agent
// Build intelligent financial tools with nim-go-sdk + Liminal banking APIs
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/becomeliminal/nim-go-sdk/core"
	"github.com/becomeliminal/nim-go-sdk/executor"
	"github.com/becomeliminal/nim-go-sdk/server"
	"github.com/becomeliminal/nim-go-sdk/tools"
	"github.com/joho/godotenv"
)

func main() {
	// ============================================================================
	// CONFIGURATION
	// ============================================================================
	_ = godotenv.Load()

	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	if anthropicKey == "" {
		log.Fatal("❌ ANTHROPIC_API_KEY environment variable is required")
	}

	liminalBaseURL := os.Getenv("LIMINAL_BASE_URL")
	if liminalBaseURL == "" {
		liminalBaseURL = "https://api.liminal.cash"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// ============================================================================
	// LIMINAL EXECUTOR SETUP
	// ============================================================================
	liminalExecutor := executor.NewHTTPExecutor(executor.HTTPExecutorConfig{
		BaseURL: liminalBaseURL,
	})
	log.Println("✅ Liminal API configured")

	// ============================================================================
	// SERVER SETUP
	// ============================================================================
	srv, err := server.New(server.Config{
		AnthropicKey:    anthropicKey,
		SystemPrompt:    hackathonSystemPrompt,
		Model:           "claude-sonnet-4-20250514",
		MaxTokens:       4096,
		LiminalExecutor: liminalExecutor,
	})
	if err != nil {
		log.Fatal(err)
	}

	// ============================================================================
	// ADD LIMINAL BANKING TOOLS
	// ============================================================================
	srv.AddTools(tools.LiminalTools(liminalExecutor)...)
	log.Println("✅ Added 9 Liminal banking tools")

	// ============================================================================
	// ADD CUSTOM TOOLS
	// ============================================================================
	srv.AddTool(createSpendingAnalyzerTool(liminalExecutor))
	log.Println("✅ Added custom spending analyzer tool")

	srv.AddTool(createCalcTradeRiskTool())
	log.Println("✅ Added custom trade risk tool (Python)")

	// ============================================================================
	// START SERVER
	// ============================================================================
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Println("🚀 Hackathon Starter Server Running")
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("📡 WebSocket endpoint: ws://localhost:%s/ws", port)
	log.Printf("💚 Health check: http://localhost:%s/health", port)
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Println("Ready for connections! Start your frontend with: cd frontend && npm run dev")
	log.Println()

	if err := srv.Run(":" + port); err != nil {
		log.Fatal(err)
	}
}

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const hackathonSystemPrompt = `You are Nim, a friendly AI financial assistant built for the Liminal Vibe Banking Hackathon.

WHAT YOU DO:
You help users manage their money using Liminal's stablecoin banking platform. You can check balances, review transactions, send money, and manage savings - all through natural conversation.

CONVERSATIONAL STYLE:
- Be friendly and conversational (not robotic)
- Use clear language and keep money-related details precise
- Ask clarifying questions when something is unclear
- Explain things simply without talking down to the user

WHEN TO USE TOOLS:
- Use tools immediately for simple queries ("what's my balance?")
- For actions, gather all required info first ("send $50 to @alice")
- Always confirm before executing money movements
- Don't use tools for general questions about how things work

MONEY MOVEMENT RULES (IMPORTANT):
- ALL money movements require explicit user confirmation
- Show a clear summary before confirming:
  * send_money: "Send $50 USD to @alice"
  * deposit_savings: "Deposit $100 USD into savings"
  * withdraw_savings: "Withdraw $50 USD from savings"
- Never assume amounts or recipients
- Always use the exact currency the user specified

AVAILABLE BANKING TOOLS:
- Check wallet balance (get_balance)
- Check savings balance and APY (get_savings_balance)
- View savings rates (get_vault_rates)
- View transaction history (get_transactions)
- Get profile info (get_profile)
- Search for users (search_users)
- Send money (send_money) - requires confirmation
- Deposit to savings (deposit_savings) - requires confirmation
- Withdraw from savings (withdraw_savings) - requires confirmation

CUSTOM ANALYTICAL TOOLS:
- Analyze spending patterns (analyze_spending)
- Calculate risk for a proposed swap/trade (calc_trade_risk)

TIPS FOR GREAT INTERACTIONS:
- Suggest relevant next steps when helpful ("Want me to compare this to last month?")
- Explain why you’re recommending something
- Keep outputs structured (numbers, bullets, clear takeaways)

Remember: You help users make safer, more informed financial decisions.`

// ============================================================================
// CUSTOM TOOL: SPENDING ANALYZER
// ============================================================================

func createSpendingAnalyzerTool(liminalExecutor core.ToolExecutor) core.Tool {
	return tools.New("analyze_spending").
		Description("Analyze the user's spending patterns over a specified time period. Returns insights about spending velocity, categories, and trends.").
		Schema(tools.ObjectSchema(map[string]interface{}{
			"days": tools.IntegerProperty("Number of days to analyze (default: 30)"),
		})).
		Handler(func(ctx context.Context, toolParams *core.ToolParams) (*core.ToolResult, error) {
			var params struct {
				Days int `json:"days"`
			}
			if err := json.Unmarshal(toolParams.Input, &params); err != nil {
				return &core.ToolResult{
					Success: false,
					Error:   fmt.Sprintf("invalid input: %v", err),
				}, nil
			}
			if params.Days == 0 {
				params.Days = 30
			}

			txRequest := map[string]interface{}{"limit": 100}
			txRequestJSON, _ := json.Marshal(txRequest)

			txResponse, err := liminalExecutor.Execute(ctx, &core.ExecuteRequest{
				UserID:    toolParams.UserID,
				Tool:      "get_transactions",
				Input:     txRequestJSON,
				RequestID: toolParams.RequestID,
			})
			if err != nil {
				return &core.ToolResult{
					Success: false,
					Error:   fmt.Sprintf("failed to fetch transactions: %v", err),
				}, nil
			}
			if !txResponse.Success {
				return &core.ToolResult{
					Success: false,
					Error:   fmt.Sprintf("transaction fetch failed: %s", txResponse.Error),
				}, nil
			}

			var transactions []map[string]interface{}
			var txData map[string]interface{}
			if err := json.Unmarshal(txResponse.Data, &txData); err == nil {
				// This depends on the exact response shape. We try a few common patterns.
				if txArray, ok := txData["transactions"].([]interface{}); ok {
					for _, tx := range txArray {
						if txMap, ok := tx.(map[string]interface{}); ok {
							transactions = append(transactions, txMap)
						}
					}
				} else if txArray, ok := txData["data"].([]interface{}); ok {
					for _, tx := range txArray {
						if txMap, ok := tx.(map[string]interface{}); ok {
							transactions = append(transactions, txMap)
						}
					}
				}
			}

			analysis := analyzeTransactions(transactions, params.Days)

			result := map[string]interface{}{
				"period_days":        params.Days,
				"total_transactions": len(transactions),
				"analysis":           analysis,
				"generated_at":       time.Now().Format(time.RFC3339),
			}

			return &core.ToolResult{
				Success: true,
				Data:    result,
			}, nil
		}).
		Build()
}

func analyzeTransactions(transactions []map[string]interface{}, days int) map[string]interface{} {
	if len(transactions) == 0 {
		return map[string]interface{}{
			"summary": "No transactions found in the specified period",
		}
	}

	var totalSpent, totalReceived float64
	var spendCount, receiveCount int

	for _, tx := range transactions {
		txType, _ := tx["type"].(string)

		// Amount can come in different types depending on API. Try to normalize.
		var amount float64
		switch v := tx["amount"].(type) {
		case float64:
			amount = v
		case int:
			amount = float64(v)
		case string:
			// Best-effort parse; ignore errors.
			var parsed float64
			_, _ = fmt.Sscanf(v, "%f", &parsed)
			amount = parsed
		}

		switch txType {
		case "send":
			totalSpent += amount
			spendCount++
		case "receive":
			totalReceived += amount
			receiveCount++
		}
	}

	avgDailySpend := 0.0
	if days > 0 {
		avgDailySpend = totalSpent / float64(days)
	}

	return map[string]interface{}{
		"total_spent":     fmt.Sprintf("%.2f", totalSpent),
		"total_received":  fmt.Sprintf("%.2f", totalReceived),
		"spend_count":     spendCount,
		"receive_count":   receiveCount,
		"avg_daily_spend": fmt.Sprintf("%.2f", avgDailySpend),
		"velocity":        calculateVelocity(spendCount, days),
		"insights": []string{
			fmt.Sprintf("You made %d spending transactions over %d days.", spendCount, days),
			fmt.Sprintf("Average daily spend: $%.2f.", avgDailySpend),
			"If you want, I can help you set a simple savings goal based on this pattern.",
		},
	}
}

func calculateVelocity(transactionCount, days int) string {
	if days <= 0 {
		return "unknown"
	}
	txPerWeek := float64(transactionCount) / float64(days) * 7

	switch {
	case txPerWeek < 2:
		return "low"
	case txPerWeek < 7:
		return "moderate"
	default:
		return "high"
	}
}

// ============================================================================
// PYTHON BRIDGE + CUSTOM TOOL: TRADE RISK
// ============================================================================

// runPythonJSON runs a Python script with JSON input (stdin) and expects JSON output (stdout).
// It will prefer .venv/bin/python if present (so your local venv is used), otherwise python3.
func runPythonJSON(ctx context.Context, scriptRelPath string, payload any) (any, error) {
	in, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	pythonBin := "python3"
	if _, err := os.Stat(".venv/bin/python"); err == nil {
		pythonBin = ".venv/bin/python"
	}

	scriptPath := filepath.Clean(scriptRelPath)
	cmd := exec.CommandContext(ctx, pythonBin, scriptPath)
	cmd.Stdin = bytes.NewReader(in)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("python failed: %v | stderr: %s", err, stderr.String())
	}

	var out any
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		return nil, fmt.Errorf("python returned invalid JSON: %v | output: %s", err, stdout.String())
	}
	return out, nil
}

func createCalcTradeRiskTool() core.Tool {
	return tools.New("calc_trade_risk").
		Description("Calculates a risk score for swapping from one token to another using our Python risk engine.").
		Schema(tools.ObjectSchema(map[string]interface{}{
			"from_token": tools.StringProperty("Token you are selling/spending (e.g., 'USDC')"),
			"to_token":   tools.StringProperty("Token you want to buy (e.g., 'BTC')"),
			"amount":     tools.NumberProperty("Amount of from_token (e.g., 1000)"),
		})).
		Handler(func(ctx context.Context, toolParams *core.ToolParams) (*core.ToolResult, error) {
			var p struct {
				FromToken string  `json:"from_token"`
				ToToken   string  `json:"to_token"`
				Amount    float64 `json:"amount"`
			}
			if err := json.Unmarshal(toolParams.Input, &p); err != nil {
				return &core.ToolResult{
					Success: false,
					Error:   fmt.Sprintf("invalid input: %v", err),
				}, nil
			}
			if p.FromToken == "" || p.ToToken == "" || p.Amount <= 0 {
				return &core.ToolResult{
					Success: false,
					Error:   "from_token, to_token, and amount are required (amount must be > 0)",
				}, nil
			}

			payload := map[string]any{
				"from_token":    p.FromToken,
				"to_token":      p.ToToken,
				"amount":        p.Amount,
				"coingecko_key": os.Getenv("COINGECKO_KEY"),
			}

			out, err := runPythonJSON(ctx, "python_tools/risk_cli.py", payload)
			if err != nil {
				return &core.ToolResult{
					Success: false,
					Error:   err.Error(),
				}, nil
			}

			return &core.ToolResult{
				Success: true,
				Data:    out,
			}, nil
		}).
		Build()
}
