// Hackathon Starter: AI Financial Agent (Risk-only swaps)
// Runs python risk_cli.py for "swap X A to B" and returns a risk report.
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

	// Where your risk_cli.py lives.
	// Set this if your python files are not in python_tools/.
	// Example: RISK_SCRIPT_PATH=/absolute/path/to/risk_cli.py
	riskScriptPath := os.Getenv("RISK_SCRIPT_PATH")
	if riskScriptPath == "" {
		riskScriptPath = "python_tools/risk_cli.py"
	}

	// ============================================================================
	// LIMINAL EXECUTOR SETUP (still useful for banking tools)
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
	// ADD LIMINAL BANKING TOOLS (optional, but fine to keep)
	// ============================================================================
	srv.AddTools(tools.LiminalTools(liminalExecutor)...)
	log.Println("✅ Added Liminal banking tools")

	// ============================================================================
	// ADD CUSTOM TOOL: calc_trade_risk (Python risk model)
	// ============================================================================
	srv.AddTool(createCalcTradeRiskTool(riskScriptPath))
	log.Println("✅ Added calc_trade_risk (Python risk model)")

	// ============================================================================
	// START SERVER
	// ============================================================================
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Println("🚀 Server Running")
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("📡 WebSocket endpoint: ws://localhost:%s/ws", port)
	log.Printf("💚 Health check: http://localhost:%s/health", port)
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Println()

	if err := srv.Run(":" + port); err != nil {
		log.Fatal(err)
	}
}

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const hackathonSystemPrompt = `You are Nim, a friendly AI financial assistant built for the Liminal Vibe Banking Hackathon.

GOAL:
When the user proposes a swap/trade like "swap 10 usdt to btc", you MUST run the risk model and return a risk report.

SWAP / TRADE BEHAVIOR (IMPORTANT):
- Do NOT ask permission.
- Do NOT check bank balance or enforce budgets.
- Immediately call calc_trade_risk with:
  from_token = the asset being spent
  to_token   = the asset being bought
  amount     = the numeric amount
- After the tool returns, present the risk report clearly.

OUTPUT:
- Keep it structured.
- If the tool returns an error, show the exact error message and suggest corrected token symbols if relevant.
- Do NOT ask the user if they want to proceed with the swap (we are only reporting risk).

MONEY MOVEMENT:
- Only require explicit confirmation for money movement tools (send_money, deposit_savings, withdraw_savings).`

// ============================================================================
// PYTHON BRIDGE
// ============================================================================

// runPythonJSON runs a Python script with JSON input (stdin) and expects JSON output (stdout).
// Prefers .venv/bin/python if present, otherwise python3.
func runPythonJSON(ctx context.Context, scriptPath string, payload any) (any, error) {
	in, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	pythonBin := "python3"
	if _, err := os.Stat(".venv/bin/python"); err == nil {
		pythonBin = ".venv/bin/python"
	}

	cleanPath := filepath.Clean(scriptPath)

	cmd := exec.CommandContext(ctx, pythonBin, cleanPath)
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

// ============================================================================
// CUSTOM TOOL: calc_trade_risk
// ============================================================================

func createCalcTradeRiskTool(riskScriptPath string) core.Tool {
	return tools.New("calc_trade_risk").
		Description("Runs the Python risk model (risk_cli.py) for a proposed swap and returns a risk report.").
		Schema(tools.ObjectSchema(map[string]interface{}{
			"from_token": tools.StringProperty("Token you are selling/spending (e.g., 'USDT')"),
			"to_token":   tools.StringProperty("Token you want to buy (e.g., 'BTC')"),
			"amount":     tools.NumberProperty("Amount of from_token (e.g., 10)"),
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

			// Payload expected by your risk_cli.py:
			// { "from_token": "...", "to_token": "...", "amount": 10, "coingecko_key": "..." }
			payload := map[string]any{
				"from_token":    p.FromToken,
				"to_token":      p.ToToken,
				"amount":        p.Amount,
				"coingecko_key": os.Getenv("COINGECKO_KEY"),
			}

			// Give Python a reasonable timeout
			rctx, cancel := context.WithTimeout(ctx, 25*time.Second)
			defer cancel()

			out, err := runPythonJSON(rctx, riskScriptPath, payload)
			if err != nil {
				return &core.ToolResult{
					Success: false,
					Error:   err.Error(),
				}, nil
			}

			// NEW: If Python returned {"error": "..."} treat it as a tool error
			if m, ok := out.(map[string]any); ok {
				if e, ok := m["error"].(string); ok && e != "" {
					return &core.ToolResult{
						Success: false,
						Error:   e,
					}, nil
				}
			}

			return &core.ToolResult{
				Success: true,
				Data:    out,
			}, nil
		}).
		Build()
}
