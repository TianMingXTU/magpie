package gateway

import (
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// A Chat API's reply that reasons and then ends with nothing said, its
// finish_reason stop and its tokens counted: catpaw's longcat-2.0
// answered 60 requests in a day with 200s that said nothing at all, and
// workbuddy's hy4free over a thousand in a week that said only their
// reasoning, and Claude Code took the turn for one answered and stopped
// its run there (#667).
var chatThoughtThenStop = sse(
	`data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"first thought"}}]}`,
	`data: {"id":"c1","choices":[{"index":0,"delta":{"reasoning_content":" second thought"}}]}`,
	`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}`,
	`data: [DONE]`)

// The same with nothing said or reasoned at all, its tokens counted too.
var chatNothingThenStop = sse(
	`data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
	`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}`,
	`data: [DONE]`)

// A Chat API's reply that says something is the answer it is.
var chatSaidThenStop = sse(
	`data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"hm"}}]}`,
	`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hello"}}]}`,
	`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}`,
	`data: [DONE]`)

// That reply cut at its length, having said nothing, is the void's failure
// (#2972) or an answer ended there, not the empty reply's (#667).
var chatThoughtThenLength = sse(
	`data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"first thought"}}]}`,
	`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}`,
	`data: [DONE]`)

// A Chat upstream's reply that says nothing — only reasoning, or nothing at
// all — is no answer, whatever it counted, as a Gemini one isn't (#667).
// Told to the agent as a turn ended as it should, it stopped Claude Code's
// run mid-task, the group's next member never asked. It is asked again
// here first, then ends with the empty reply's error in the client's own
// protocol.
func TestChatEmptyReplyIsAnError(t *testing.T) {
	f := &fake{t: t}
	setup(t, provider.Chat, f)
	for what, reply := range map[string]string{
		"only reasoning": chatThoughtThenStop,
		"nothing":        chatNothingThenStop,
	} {
		f.reply = reply
		for _, a := range []struct{ name, path, body, ended string }{
			{"messages", "/v1/messages", `{"model":"m1","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`, `"end_turn"`},
			{"responses", "/v1/responses", `{"model":"m1","stream":true,"input":"hi"}`, "response.completed"},
		} {
			f.calls = 0
			code, body := post(t, a.path, a.body)
			if strings.Contains(body, a.ended) || !strings.Contains(body, "an empty reply") {
				t.Errorf("%s, %s: a normal end, not an error: %d %s", what, a.name, code, body)
			}
			if f.calls < 2 {
				t.Errorf("%s, %s: asked %d times, not again", what, a.name, f.calls)
			}
		}
		code, body := post(t, "/v1/messages", `{"model":"m1","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
		if code != 502 || !strings.Contains(body, "an empty reply") {
			t.Errorf("%s, not streamed: %d %s", what, code, body)
		}
	}

	// what says something goes as it came, and a reply cut at its length
	// is the void's or the length's (#2972), not the empty reply's
	for what, reply := range map[string]string{
		"text":              chatSaidThenStop,
		"cut at its length": chatThoughtThenLength,
	} {
		f.reply = reply
		code, body := post(t, "/v1/messages", `{"model":"m1","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
		if strings.Contains(body, "an empty reply") {
			t.Errorf("%s: %d %s", what, code, body)
		}
		if what == "text" && (code != 200 || !strings.Contains(body, "hello") || !strings.Contains(body, `"stop_reason":"end_turn"`)) {
			t.Errorf("%s: %d %s", what, code, body)
		}
	}
}

// With another member behind it, the reasoning the agent already has
// stays and that member's answer follows it in the same stream, as one
// after a refusal does (handOverEmpty, refusalHandover): Claude Code and
// Codex are not told the turn ended with nothing said, and the member is
// asked as any empty reply's is (#667, #1221).
func TestChatEmptyReplyIsHandedToTheNext(t *testing.T) {
	for _, client := range []struct {
		name, path, ask string
		check           func(t *testing.T, body string)
	}{
		{"messages", "/v1/messages", claudeCodeAsk, checkAnthropicSpliced},
		{"responses", "/v1/responses", codexAsk, checkResponsesSpliced},
	} {
		t.Run(client.name, func(t *testing.T) {
			fresh(t)
			a := &scripted{replies: []reply{{200, "text/event-stream", chatThoughtThenStop}}}
			b := &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}}
			scriptedOn(t, "a", provider.Chat, a)
			scriptedOn(t, "b", provider.Anthropic, b)
			refusalGroup(t, "a/m", "b/m")
			s := New()
			code, body := sendTo(s, client.path, client.ask)
			if code != 200 || a.n != 2 || b.n != 1 {
				t.Fatalf("%d (a %d, b %d): %s", code, a.n, b.n, body)
			}
			if !strings.Contains(body, "first thought") || !strings.Contains(body, " second thought") || !strings.Contains(body, "from b") {
				t.Fatalf("not the reasoning and then the next member's answer: %s", body)
			}
			if strings.Contains(body, "an empty reply") {
				t.Fatalf("the empty reply reached the agent: %s", body)
			}
			client.check(t, body)
			r := lastRoute(s)
			if len(r.Tries) != 2 || r.Tries[0].Fail != failOther || r.Tries[0].Rest != nil || r.Tries[1].Fail != "" {
				t.Fatalf("tries: %+v", r.Tries)
			}
			if s.resting("a") {
				t.Fatal("the member that answered nothing was set aside")
			}
		})
	}
}

// A member whose reasoning was held for a refusal after it (holdsThinking,
// #248) passes none of it when it ends empty: nothing of the try reached
// the agent, so its failure is asked of the next member as any held one
// is, whose answer is the first thing the agent sees.
func TestChatEmptyReplyAfterHeldThinkingAsksTheNext(t *testing.T) {
	for _, client := range []struct {
		name, path, ask string
	}{
		{"messages", "/v1/messages", claudeCodeAsk},
		{"responses", "/v1/responses", codexAsk},
	} {
		t.Run(client.name, func(t *testing.T) {
			fresh(t)
			a := &scripted{replies: []reply{{200, "text/event-stream", chatThoughtThenStop}}}
			b := &scripted{replies: []reply{{200, "text/event-stream", anthropicAnswer}}}
			scriptedOn(t, "a", provider.Chat, a)
			pa, err := provider.Find("a")
			if err != nil {
				t.Fatal(err)
			}
			pa.Models = []string{"claude-x"}
			if err := provider.Save(*pa); err != nil {
				t.Fatal(err)
			}
			scriptedOn(t, "b", provider.Anthropic, b)
			refusalGroup(t, "a/claude-x", "b/m")
			s := New()
			code, body := sendTo(s, client.path, client.ask)
			if code != 200 || a.n != 2 || b.n != 1 || !strings.Contains(body, "from b") {
				t.Fatalf("%d (a %d, b %d): %s", code, a.n, b.n, body)
			}
			if strings.Contains(body, "first thought") || strings.Contains(body, "an empty reply") {
				t.Fatalf("what was held reached the agent: %s", body)
			}
			if client.name == "messages" {
				if strings.Count(body, `"type":"message_start"`) != 1 {
					t.Fatalf("not one message: %s", body)
				}
			} else if strings.Count(body, `"type":"response.created"`) != 1 {
				t.Fatalf("not one response: %s", body)
			}
			r := lastRoute(s)
			if len(r.Tries) != 2 || r.Tries[0].Rest != nil || r.Tries[1].Fail != "" {
				t.Fatalf("tries: %+v", r.Tries)
			}
		})
	}
}
