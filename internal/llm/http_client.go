package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/akhenakh/qmd/internal/util"
)

type HTTPClient struct {
	BaseURL    string
	Model      string
	TargetDim  int
	APIKey     string
	HTTPClient *http.Client
}

// NewHTTPClient builds a client for an embedding backend.
//
// If apiKey is empty, requests go to Ollama's native /api/embeddings
// endpoint (no auth), preserving the original behavior.
//
// If apiKey is non-empty, requests go to the OpenAI-compatible
// /v1/embeddings endpoint with an Authorization: Bearer header instead —
// this is what's needed to talk to a LiteLLM proxy (or any other
// OpenAI-compatible embeddings endpoint), since LiteLLM does not expose
// Ollama's native /api/embeddings route.
func NewHTTPClient(baseURL, model string, targetDim int, apiKey string) *HTTPClient {
	return &HTTPClient{
		BaseURL:   baseURL,
		Model:     model,
		TargetDim: targetDim,
		APIKey:    apiKey,
		HTTPClient: &http.Client{
			// Increased timeout to 5 minutes for large contexts/slow models
			Timeout: 300 * time.Second,
		},
	}
}

// EmbedRequest follows Ollama's native API format
type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// OpenAIEmbedRequest follows the OpenAI-compatible API format (used by
// LiteLLM and similar proxies).
type OpenAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type OpenAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (c *HTTPClient) Embed(text string, isQuery bool) ([]float32, error) {
	// Simple Nomic/Gemma formatting logic
	prefix := "search_document: "
	if isQuery {
		prefix = "search_query: "
	}
	prompt := prefix + text

	var vec []float32
	if c.APIKey != "" {
		v, err := c.embedOpenAI(prompt)
		if err != nil {
			return nil, err
		}
		vec = v
	} else {
		v, err := c.embedOllama(prompt)
		if err != nil {
			return nil, err
		}
		vec = v
	}

	// Handle Matryoshka Truncation
	if c.TargetDim > 0 && len(vec) > c.TargetDim {
		util.Debug("LLM [HTTP] Truncating vector from %d to %d", len(vec), c.TargetDim)
		vec = vec[:c.TargetDim]

		// Re-normalize after truncation
		var sum float64
		for _, v := range vec {
			sum += float64(v * v)
		}
		sum = math.Sqrt(sum)
		if sum > 0 {
			norm := float32(1.0 / sum)
			for i := range vec {
				vec[i] *= norm
			}
		}
	}

	return vec, nil
}

func (c *HTTPClient) embedOllama(prompt string) ([]float32, error) {
	reqBody := EmbedRequest{
		Model:  c.Model,
		Prompt: prompt,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	// Log Raw Request
	util.Debug("LLM [HTTP] Request Payload:\n%s", string(jsonData))

	resp, err := c.HTTPClient.Post(c.BaseURL+"/api/embeddings", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		util.Debug("LLM [HTTP] Connection Error: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		util.Debug("LLM [HTTP] API Status Error: %s", resp.Status)
		return nil, fmt.Errorf("API returned status: %s", resp.Status)
	}

	// Read Body for Logging
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Log Raw Response
	// Note: This can be very large due to vector arrays
	util.Debug("LLM [HTTP] Response Payload:\n%s", string(bodyBytes))

	// Decode
	var result EmbedResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, err
	}

	return result.Embedding, nil
}

func (c *HTTPClient) embedOpenAI(prompt string) ([]float32, error) {
	reqBody := OpenAIEmbedRequest{
		Model: c.Model,
		Input: []string{prompt},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	util.Debug("LLM [OpenAI] Request Payload:\n%s", string(jsonData))

	req, err := http.NewRequest(http.MethodPost, c.BaseURL+"/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		util.Debug("LLM [OpenAI] Connection Error: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		util.Debug("LLM [OpenAI] API Status Error: %s\n%s", resp.Status, string(bodyBytes))
		return nil, fmt.Errorf("API returned status: %s: %s", resp.Status, string(bodyBytes))
	}

	util.Debug("LLM [OpenAI] Response Payload:\n%s", string(bodyBytes))

	var result OpenAIEmbedResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, err
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("OpenAI-compatible API returned no embedding data")
	}

	return result.Data[0].Embedding, nil
}

func (c *HTTPClient) Close() error {
	// HTTP client doesn't need specific cleanup
	return nil
}
