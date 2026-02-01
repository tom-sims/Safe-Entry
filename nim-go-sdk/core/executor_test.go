package core

import (
	"encoding/json"
	"testing"
)

func TestExecutorTool_GetSummary(t *testing.T) {
	tests := []struct {
		name            string
		summaryTemplate string
		input           map[string]interface{}
		want            string
	}{
		{
			name:            "send_money template",
			summaryTemplate: "Send {{.amount}} {{.currency}} to {{.recipient}}",
			input: map[string]interface{}{
				"amount":    "50.00",
				"currency":  "USD",
				"recipient": "@alice",
			},
			want: "Send 50.00 USD to @alice",
		},
		{
			name:            "execute_contract_call template",
			summaryTemplate: "Execute contract call on chain {{.chain_id}} to {{.to}}",
			input: map[string]interface{}{
				"chain_id": 42161,
				"to":       "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
			},
			want: "Execute contract call on chain 42161 to 0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
		},
		{
			name:            "deposit_savings template",
			summaryTemplate: "Deposit {{.amount}} {{.currency}} into savings",
			input: map[string]interface{}{
				"amount":   "100",
				"currency": "USD",
			},
			want: "Deposit 100 USD into savings",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := NewExecutorTool(ToolDefinition{
				ToolName:        "test_tool",
				SummaryTemplate: tt.summaryTemplate,
			}, nil)

			inputBytes, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("Failed to marshal input: %v", err)
			}

			got := tool.GetSummary(inputBytes)
			if got != tt.want {
				t.Errorf("GetSummary() = %q, want %q", got, tt.want)
			}
		})
	}
}
